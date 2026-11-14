"""MCP server with in-process query embedder for Vision Memory Book."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer as FastMCP

from vision_memory_book.ai.base import AIProvider
from vision_memory_book.config import Settings, get_settings
from vision_memory_book.db.connection import get_connection
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class MCPContext:
    """Dependencies wired into the Vision Memory Book MCP server."""

    repo: Repository
    queue: JobQueue
    embed_provider: AIProvider | None = None


def create_mcp_server(
    repo: Repository,
    queue: JobQueue,
    embed_provider: AIProvider | None = None,
) -> FastMCP:
    """Create and configure the Vision Memory Book MCP server."""
    ctx = MCPContext(repo=repo, queue=queue, embed_provider=embed_provider)
    mcp = FastMCP("vision-memory-book")

    @mcp.tool()
    async def search_photos(
        query: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        year: int | None = None,
        tags: list[str] | None = None,
        group_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search photos using hybrid semantic vector similarity, FTS5 keywords, and SQL filters."""
        query_embedding: list[float] | None = None
        clean_query = query.strip() if query is not None else None
        if clean_query and ctx.embed_provider is not None:
            try:
                vectors = await ctx.embed_provider.embed([clean_query])
                if vectors:
                    query_embedding = vectors[0]
            except Exception as exc:
                logger.warning(
                    "In-process query embedding failed; falling back to FTS5 search: %s",
                    exc,
                )

        return ctx.repo.search_photos(
            query_embedding=query_embedding,
            text_query=clean_query,
            start_date=start_date,
            end_date=end_date,
            year=year,
            tags=tags,
            group_id=group_id,
            limit=limit,
            offset=offset,
        )

    @mcp.tool()
    async def get_photo_details(photo_id: int) -> dict[str, Any] | None:
        """Return full metadata, tags, description, and group memberships for a photo."""
        return ctx.repo.get_photo(photo_id)

    @mcp.tool()
    async def get_year_stats(year: int) -> dict[str, Any]:
        """Return photo counts by status, by month, and top tags for a given year."""
        return ctx.repo.get_year_stats(year)

    @mcp.tool()
    async def get_queue_status() -> dict[str, Any]:
        """Return job queue counts by job_type and status, plus recent errors."""
        return ctx.queue.stats()

    @mcp.tool()
    async def control_queue(
        action: str,
        job_type: str | None = None,
    ) -> dict[str, Any]:
        """Control the background job queue ('pause', 'resume', or 'retry_failed')."""
        if action == "pause":
            affected = ctx.queue.pause_all(job_type=job_type)
        elif action == "resume":
            affected = ctx.queue.resume_all(job_type=job_type)
        elif action == "retry_failed":
            affected = ctx.queue.retry_failed(job_type=job_type)
        else:
            raise ValueError(
                f"Unsupported queue action {action!r}; expected 'pause', 'resume', or 'retry_failed'."
            )
        return {
            "action": action,
            "job_type": job_type,
            "affected": affected,
            "queue": ctx.queue.stats(),
        }

    @mcp.tool()
    async def reprocess_photo(photo_id: int) -> dict[str, Any]:
        """Enqueue a fresh vision indexing job for an existing photo."""
        photo = ctx.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Photo {photo_id} not found")
        job_id = ctx.queue.enqueue("vision", photo_id)
        return {
            "photo_id": photo_id,
            "job_id": job_id,
            "job_type": "vision",
            "status": "pending",
        }

    @mcp.tool()
    async def list_groups(year: int | None = None) -> list[dict[str, Any]]:
        """List photo groups, optionally filtered by year."""
        return ctx.repo.list_groups(year=year)

    @mcp.tool()
    async def create_group(
        name: str,
        year: int | None = None,
        description: str | None = None,
        kind: str = "manual",
    ) -> dict[str, Any]:
        """Create a new photo group and return its full record."""
        group_id = ctx.repo.create_group(
            name=name,
            year=year,
            description=description,
            kind=kind,
        )
        group = ctx.repo.get_group(group_id)
        assert group is not None
        return group

    @mcp.tool()
    async def add_photos_to_group(
        group_id: int,
        photo_ids: list[int],
        captions: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Add photos to a group with optional per-photo captions (keyed by photo_id string)."""
        int_captions: dict[int, str] | None = (
            {int(k): v for k, v in captions.items()} if captions else None
        )
        added = ctx.repo.add_photos_to_group(
            group_id=group_id,
            photo_ids=photo_ids,
            captions=int_captions,
        )
        return {
            "group_id": group_id,
            "added": added,
            "photo_ids": photo_ids,
        }

    @mcp.tool()
    async def remove_photos_from_group(
        group_id: int,
        photo_ids: list[int],
    ) -> dict[str, Any]:
        """Remove photos from a group."""
        removed = ctx.repo.remove_photos_from_group(
            group_id=group_id,
            photo_ids=photo_ids,
        )
        return {
            "group_id": group_id,
            "removed": removed,
            "photo_ids": photo_ids,
        }

    @mcp.tool()
    async def reorder_group(
        group_id: int,
        ordered_photo_ids: list[int],
    ) -> dict[str, Any]:
        """Update the sort order of photos within a group."""
        ctx.repo.reorder_group(
            group_id=group_id,
            ordered_photo_ids=ordered_photo_ids,
        )
        return {
            "group_id": group_id,
            "ordered_photo_ids": ordered_photo_ids,
            "ok": True,
        }

    @mcp.tool()
    async def get_group_photos(group_id: int) -> list[dict[str, Any]]:
        """Return all photos in a group ordered by sort_order."""
        return ctx.repo.get_group_photos(group_id)

    @mcp.tool()
    async def export_repick_manifest(group_id: int) -> list[dict[str, Any]]:
        """Export checklist of photos in a group for Final-Touch (=d) Google Photos re-pick."""
        return ctx.repo.export_repick_manifest(group_id)

    @mcp.tool()
    async def save_book_draft(
        year: int,
        title: str,
        chapters_json: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist a structured memory book outline draft and return the saved book record."""
        book_id = ctx.repo.save_book(
            year=year,
            title=title,
            outline_json={"chapters": chapters_json},
        )
        book = ctx.repo.get_book(book_id)
        assert book is not None
        return book

    return mcp


def _build_embed_provider(settings: Settings) -> AIProvider | None:
    """Instantiate OpenAICompatProvider for query embeddings when available."""
    try:
        from vision_memory_book.ai.openai_compat import OpenAICompatProvider
    except ImportError:
        logger.warning("OpenAICompatProvider not available; query embedding disabled.")
        return None

    try:
        return OpenAICompatProvider(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
            embed_model=settings.embed_model,
        )
    except Exception as exc:
        logger.warning("Failed to initialize OpenAICompatProvider: %s", exc)
        return None


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the Vision Memory Book MCP server."""
    parser = argparse.ArgumentParser(description="Vision Memory Book MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    conn = get_connection(settings.db_path)
    try:
        repo = Repository(conn)
        queue = JobQueue(conn)
        embed_provider = _build_embed_provider(settings)
        server = create_mcp_server(repo=repo, queue=queue, embed_provider=embed_provider)
        server.run(transport=args.transport)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
