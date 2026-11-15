"""Command-line interface (`vmb`) for Vision Memory Book."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import time
from typing import Sequence
import webbrowser

import uvicorn

from vision_memory_book.ai.openai_compat import build_ai_providers
from vision_memory_book.book.generator import MemoryBookGenerator
from vision_memory_book.book.renderer import BookRenderer
from vision_memory_book.config import Settings, get_settings
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.mcp_server import server as mcp_server_mod
from vision_memory_book.picker.auth import GoogleOAuthManager
from vision_memory_book.picker.client import PickerClient, ingest_picker_session
from vision_memory_book.web.app import create_app
from vision_memory_book.workers.downloader import DownloadWorker
from vision_memory_book.workers.indexer import VisionIndexerWorker

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmb",
        description="Vision Memory Book: local-first photo indexer, MCP server, and book generator",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-db
    subparsers.add_parser(
        "init-db",
        help="Initialize SQLite database schema and local data directories",
    )

    # pick
    pick_parser = subparsers.add_parser(
        "pick",
        help="Authenticate with Google Photos Picker, create a session, and ingest picked items",
    )
    pick_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between Picker session status polls (default: 2.0)",
    )
    pick_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for user selection in Picker (default: 300.0)",
    )
    pick_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the Picker URI in a browser",
    )

    # worker
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run download and/or vision indexing background workers",
    )
    worker_parser.add_argument(
        "--type",
        dest="worker_type",
        choices=["download", "vision", "all"],
        default="all",
        help="Worker type to run (default: all)",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="Process available jobs once and exit",
    )
    worker_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds when idle (default: 2.0)",
    )

    # mcp
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run the Vision Memory Book MCP server",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )

    # web
    web_parser = subparsers.add_parser(
        "web",
        help="Start the local FastAPI Web UI",
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    # book
    book_parser = subparsers.add_parser(
        "book",
        help="Generate a yearly memory book (and optionally render HTML/PDF)",
    )
    book_parser.add_argument("--year", type=int, required=True, help="Target year for the book")
    book_parser.add_argument("--title", default=None, help="Optional custom book title")
    book_parser.add_argument(
        "--max-chapters",
        type=int,
        default=6,
        help="Maximum number of chapters (default: 6)",
    )
    book_parser.add_argument(
        "--max-photos-per-chapter",
        type=int,
        default=12,
        help="Maximum photos per chapter (default: 12)",
    )
    book_parser.add_argument(
        "--render",
        action="store_true",
        help="Render HTML and PDF outputs after generating the book",
    )
    book_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for rendered HTML/PDF (default: <data_dir>/books)",
    )

    return parser


def _cmd_init_db(settings: Settings) -> int:
    settings.ensure_dirs()
    conn = init_db(settings.db_path, embed_dim=settings.embed_dim)
    conn.close()
    print(f"Initialized database at {settings.db_path}")
    return 0


async def _cmd_pick(
    settings: Settings,
    poll_interval: float,
    timeout: float,
    no_browser: bool,
) -> int:
    settings.ensure_dirs()
    conn = init_db(settings.db_path, embed_dim=settings.embed_dim)
    auth = GoogleOAuthManager(settings)
    try:
        await auth.get_access_token()
    except Exception:
        await auth.authorize_local_server(open_browser=not no_browser)

    repo = Repository(conn)
    queue = JobQueue(conn)

    try:
        async with PickerClient(auth_manager=auth) as client:
            session = await client.create_session()
            session_id = str(session["id"])
            picker_uri = str(session.get("pickerUri") or "")
            expire_time = session.get("expireTime")
            repo.create_picker_session(
                session_id=session_id,
                picker_uri=picker_uri,
                expire_time=expire_time,
                status="pending",
            )
            print(f"Created Picker session: {session_id}")
            if picker_uri:
                print(f"Open Picker URL: {picker_uri}")
                if not no_browser:
                    webbrowser.open(picker_uri)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                current = await client.get_session(session_id)
                if current.get("mediaItemsSet"):
                    repo.update_picker_session_status(session_id, "completed")
                    stats = await ingest_picker_session(client, repo, queue, session_id)
                    print(
                        f"Ingested session {session_id}: "
                        f"total={stats['total']} enqueued={stats['enqueued']} skipped={stats['skipped']}"
                    )
                    return 0
                await asyncio.sleep(poll_interval)

            repo.update_picker_session_status(session_id, "expired")
            print(f"Timed out waiting for Picker session {session_id}")
            return 1
    finally:
        conn.close()


async def _cmd_worker(
    settings: Settings,
    worker_type: str,
    once: bool,
    poll_interval: float,
) -> int:
    settings.ensure_dirs()
    conn = init_db(settings.db_path, embed_dim=settings.embed_dim)
    repo = Repository(conn)
    queue = JobQueue(conn)

    auth = GoogleOAuthManager(settings)
    picker_client = PickerClient(auth_manager=auth)
    providers = build_ai_providers(settings)

    downloader = DownloadWorker(
        repo=repo,
        queue=queue,
        picker_client=picker_client,
        settings=settings,
    )
    indexer = VisionIndexerWorker(
        queue=queue,
        repo=repo,
        vision_provider=providers["vision"],
        embed_provider=providers["embed"],
    )

    try:
        if once:
            if worker_type in ("download", "all"):
                while await downloader.run_once():
                    pass
            if worker_type in ("vision", "all"):
                while await indexer.run_once():
                    pass
            return 0

        tasks: list[asyncio.Task[None]] = []
        if worker_type in ("download", "all"):
            tasks.append(
                asyncio.create_task(downloader.run_forever(poll_interval=poll_interval))
            )
        if worker_type in ("vision", "all"):
            tasks.append(
                asyncio.create_task(indexer.run_forever(poll_interval=poll_interval))
            )
        await asyncio.gather(*tasks)
        return 0
    finally:
        await picker_client.aclose()
        for provider in providers.values():
            await provider.aclose()
        conn.close()


async def _cmd_book(
    settings: Settings,
    year: int,
    title: str | None,
    max_chapters: int,
    max_photos_per_chapter: int,
    render: bool,
    output_dir: Path | None,
) -> int:
    settings.ensure_dirs()
    conn = init_db(settings.db_path, embed_dim=settings.embed_dim)
    repo = Repository(conn)
    providers = build_ai_providers(settings)

    try:
        generator = MemoryBookGenerator(repo=repo, text_provider=providers["text"])
        book = await generator.generate_year_book(
            year=year,
            title=title,
            max_chapters=max_chapters,
            max_photos_per_chapter=max_photos_per_chapter,
        )
        print(f"Generated book #{book['id']}: {book['title']} ({book['year']})")

        if render:
            target_dir = output_dir or (settings.data_dir / "books")
            renderer = BookRenderer(output_dir=target_dir)
            book = renderer.render(book, repo)
            print(f"Rendered HTML: {book.get('html_path')}")
            print(f"Rendered PDF:  {book.get('pdf_path')}")
        return 0
    finally:
        for provider in providers.values():
            await provider.aclose()
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()

    if args.command == "init-db":
        return _cmd_init_db(settings)

    if args.command == "pick":
        return asyncio.run(
            _cmd_pick(
                settings=settings,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                no_browser=args.no_browser,
            )
        )

    if args.command == "worker":
        return asyncio.run(
            _cmd_worker(
                settings=settings,
                worker_type=args.worker_type,
                once=args.once,
                poll_interval=args.poll_interval,
            )
        )

    if args.command == "mcp":
        mcp_server_mod.main(["--transport", args.transport])
        return 0

    if args.command == "web":
        app = create_app(settings=settings)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "book":
        return asyncio.run(
            _cmd_book(
                settings=settings,
                year=args.year,
                title=args.title,
                max_chapters=args.max_chapters,
                max_photos_per_chapter=args.max_photos_per_chapter,
                render=args.render,
                output_dir=args.output_dir,
            )
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
