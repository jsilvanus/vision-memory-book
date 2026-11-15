"""Tests for MemoryBookGenerator, BookRenderer, and CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from vision_memory_book.ai.base import VisionResult
from vision_memory_book.book.generator import MemoryBookGenerator
from vision_memory_book.book.renderer import BookRenderer
from vision_memory_book.cli import main as cli_main
from vision_memory_book.config import get_settings
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.repository import Repository


class _MockTextProvider:
    def __init__(self, response: str | Exception) -> None:
        self._response = response

    async def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str | None = None,
    ) -> VisionResult:
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def chat(self, messages: list[dict]) -> str:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _make_test_image(path: Path, color: tuple[int, int, int], fmt: str = "WEBP") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), color=color)
    img.save(path, format=fmt)


def _seed_year_photos(repo: Repository, tmp_path: Path, year: int = 2025) -> list[int]:
    photo_specs = [
        ("pk-1", f"{year}-01-15T10:00:00Z", "snow_hike.jpg", "Snowy mountain trail", ["winter", "hike"], (30, 80, 160)),
        ("pk-2", f"{year}-06-21T18:30:00Z", "beach_sunset.jpg", "Golden sunset on the beach", ["summer", "beach"], (220, 140, 40)),
        ("pk-3", f"{year}-10-05T14:15:00Z", "autumn_park.jpg", "Red maple trees in the park", ["autumn", "park"], (180, 60, 40)),
    ]
    ids: list[int] = []
    for idx, (picker_id, ctime, fname, desc, tags, color) in enumerate(photo_specs, start=1):
        pid = repo.upsert_photo(
            picker_id=picker_id,
            filename=fname,
            mime_type="image/jpeg",
            create_time=ctime,
            width=320,
            height=240,
        )
        cache_file = tmp_path / "cache" / str(year) / f"{pid}.webp"
        _make_test_image(cache_file, color=color, fmt="WEBP")
        repo.mark_downloaded(pid, cache_path=str(cache_file), width=320, height=240)
        repo.save_vision_index(
            photo_id=pid,
            description=desc,
            tags=tags,
            vlm_json={"description": desc, "tags": tags},
            vlm_model="test-vlm",
            embed_model="test-embed",
            embedding=[float(idx) * 0.1] * 8,
        )
        ids.append(pid)
    return ids


async def test_generate_year_book_with_fenced_json(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db", embed_dim=8)
    try:
        repo = Repository(conn)
        pids = _seed_year_photos(repo, tmp_path, year=2025)

        llm_payload = {
            "title": "2025 Adventures",
            "subtitle": "Four seasons in pictures",
            "intro": "A look back at our favorite moments.",
            "chapters": [
                {
                    "title": "Winter & Summer",
                    "intro": "From snowy peaks to warm shores.",
                    "photo_ids": [pids[0], pids[1]],
                    "captions": {
                        str(pids[0]): "Crisp January hike",
                        str(pids[1]): "Solstice sunset",
                    },
                },
                {
                    "title": "Autumn Colors",
                    "intro": "Quiet walks under maple leaves.",
                    "photo_ids": [pids[2]],
                    "captions": {str(pids[2]): "October afternoon"},
                },
            ],
        }
        fenced = f"```json\n{json.dumps(llm_payload)}\n```"
        gen = MemoryBookGenerator(repo=repo, text_provider=_MockTextProvider(fenced))

        book = await gen.generate_year_book(year=2025)
        assert book["year"] == 2025
        assert book["title"] == "2025 Adventures"

        outline = json.loads(book["outline_json"])
        assert len(outline["chapters"]) == 2
        assert outline["chapters"][0]["photo_ids"] == [pids[0], pids[1]]

        groups = repo.list_groups(year=2025)
        chapter_groups = [g for g in groups if g["kind"] == "book_chapter"]
        assert len(chapter_groups) == 2

        g0_photos = repo.get_group_photos(outline["chapters"][0]["group_id"])
        assert [p["id"] for p in g0_photos] == [pids[0], pids[1]]
        assert g0_photos[0]["caption"] == "Crisp January hike"
    finally:
        conn.close()


async def test_generate_year_book_fallback_on_invalid_json(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db", embed_dim=8)
    try:
        repo = Repository(conn)
        pids = _seed_year_photos(repo, tmp_path, year=2025)

        gen = MemoryBookGenerator(
            repo=repo,
            text_provider=_MockTextProvider("not valid json at all {{{"),
        )
        book = await gen.generate_year_book(year=2025, title="Fallback 2025")
        assert book["title"] == "Fallback 2025"

        outline = json.loads(book["outline_json"])
        assert len(outline["chapters"]) == 3
        assert outline["chapters"][0]["title"] == "January 2025"
        assert outline["chapters"][0]["photo_ids"] == [pids[0]]

        groups = repo.list_groups(year=2025)
        assert len([g for g in groups if g["kind"] == "book_chapter"]) == 3
    finally:
        conn.close()


async def test_book_renderer_creates_html_pdf_and_prefers_original(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db", embed_dim=8)
    try:
        repo = Repository(conn)
        pids = _seed_year_photos(repo, tmp_path, year=2025)

        # Give photo 1 an existing original_path (=d re-pick)
        orig_file = tmp_path / "originals" / "2025" / "snow_hike_fullres.jpg"
        _make_test_image(orig_file, color=(255, 255, 255), fmt="JPEG")
        repo.mark_original_downloaded(pids[0], str(orig_file))

        # Give photo 2 a non-existent original_path so renderer must fall back to cache_path
        repo.mark_original_downloaded(pids[1], str(tmp_path / "originals" / "missing.jpg"))

        gen = MemoryBookGenerator(
            repo=repo,
            text_provider=_MockTextProvider("invalid"),
        )
        book = await gen.generate_year_book(year=2025, title="Rendered Book")

        renderer = BookRenderer(output_dir=tmp_path / "books")
        updated = renderer.render(book, repo)

        html_path = Path(updated["html_path"])
        pdf_path = Path(updated["pdf_path"])
        assert html_path.is_file()
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")

        html_text = html_path.read_text(encoding="utf-8")
        assert "Rendered Book" in html_text
        # Photo 1 used original_path
        assert orig_file.resolve().as_uri() in html_text
        # Photo 2 fell back to cache_path
        p2 = repo.get_photo(pids[1])
        assert p2 is not None
        assert Path(p2["cache_path"]).resolve().as_uri() in html_text
    finally:
        conn.close()


def test_cli_init_db_and_help(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    db_file = tmp_path / "cli_data" / "vmb.db"
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cli_data"))
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cli_data" / "cache"))
    monkeypatch.setenv("ORIGINALS_DIR", str(tmp_path / "cli_data" / "originals"))
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "cli_data" / "token.json"))
    monkeypatch.setenv("EMBED_DIM", "8")

    try:
        rc = cli_main(["init-db"])
        assert rc == 0
        assert db_file.is_file()
    finally:
        get_settings.cache_clear()

    proc = subprocess.run(
        [sys.executable, "-m", "vision_memory_book.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for subcmd in ("init-db", "pick", "worker", "mcp", "web", "book"):
        assert subcmd in proc.stdout
