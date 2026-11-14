"""Tests for the Vision Memory Book MCP server and in-process query embedder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vision_memory_book.ai.base import VisionResult
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.mcp_server.server import create_mcp_server


def _unit_vec(dim: int, hot_index: int) -> list[float]:
    vec = [0.0] * dim
    vec[hot_index] = 1.0
    return vec


class MockEmbedProvider:
    """Mock AIProvider for testing in-process query embedding."""

    def __init__(self, vectors_by_query: dict[str, list[float]] | None = None, fail: bool = False) -> None:
        self.vectors_by_query = vectors_by_query or {}
        self.fail = fail
        self.calls: list[list[str]] = []

    async def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str | None = None,
    ) -> VisionResult:
        return VisionResult(description="mock", tags=["mock"])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("embedding service unavailable")
        result: list[list[float]] = []
        for text in texts:
            result.append(self.vectors_by_query.get(text, _unit_vec(8, 0)))
        return result

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        return "mock response"


def _extract_result(call_result: Any) -> Any:
    """Extract structured or JSON-parsed result from FastMCP/MCPServer call_tool."""
    if getattr(call_result, "structured_content", None) is not None:
        sc = call_result.structured_content
        if isinstance(sc, dict) and "result" in sc:
            return sc["result"]
        return sc
    content = getattr(call_result, "content", [])
    if not content:
        return None
    if len(content) == 1:
        return json.loads(content[0].text)
    return [json.loads(item.text) for item in content]


@pytest.fixture()
def db_env(tmp_path: Path):
    conn = init_db(tmp_path / "test.db", embed_dim=8)
    repo = Repository(conn)
    queue = JobQueue(conn)
    yield conn, repo, queue
    conn.close()


def _seed_photos(repo: Repository) -> tuple[int, int, int]:
    p1 = repo.upsert_photo(
        picker_id="pick-1",
        filename="beach_sunset.jpg",
        mime_type="image/jpeg",
        create_time="2024-07-15T18:30:00Z",
        width=2048,
        height=1365,
    )
    repo.mark_downloaded(p1, "data/cache/2024/1.webp")
    repo.save_vision_index(
        photo_id=p1,
        description="Golden sunset over ocean waves on a sandy beach",
        tags=["beach", "sunset", "ocean"],
        vlm_json={"scene": "coast"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=_unit_vec(8, 1),
    )

    p2 = repo.upsert_photo(
        picker_id="pick-2",
        filename="mountain_hike.jpg",
        mime_type="image/jpeg",
        create_time="2024-08-03T11:15:00Z",
        width=2048,
        height=1365,
    )
    repo.mark_downloaded(p2, "data/cache/2024/2.webp")
    repo.save_vision_index(
        photo_id=p2,
        description="Alpine trail with pine trees and snowy peaks",
        tags=["mountain", "hike", "nature"],
        vlm_json={"scene": "mountain"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=_unit_vec(8, 2),
    )

    p3 = repo.upsert_photo(
        picker_id="pick-3",
        filename="winter_cabin.jpg",
        mime_type="image/jpeg",
        create_time="2023-12-20T09:00:00Z",
        width=1024,
        height=768,
    )
    repo.mark_downloaded(p3, "data/cache/2023/3.webp")
    repo.save_vision_index(
        photo_id=p3,
        description="Cozy wooden cabin surrounded by fresh snow",
        tags=["winter", "cabin", "snow"],
        vlm_json={"scene": "winter"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=_unit_vec(8, 3),
    )
    return p1, p2, p3


async def test_all_tools_registered(db_env) -> None:
    _, repo, queue = db_env
    server = create_mcp_server(repo, queue, MockEmbedProvider())
    tools = await server.list_tools()
    names = {t.name for t in tools}
    expected = {
        "search_photos",
        "get_photo_details",
        "get_year_stats",
        "get_queue_status",
        "control_queue",
        "reprocess_photo",
        "list_groups",
        "create_group",
        "add_photos_to_group",
        "remove_photos_from_group",
        "reorder_group",
        "get_group_photos",
        "export_repick_manifest",
        "save_book_draft",
    }
    assert expected.issubset(names)


async def test_search_photos_hybrid_and_filters(db_env) -> None:
    _, repo, queue = db_env
    p1, p2, p3 = _seed_photos(repo)
    provider = MockEmbedProvider(
        vectors_by_query={
            "ocean sunset": _unit_vec(8, 1),
            "mountain": _unit_vec(8, 2),
        }
    )
    server = create_mcp_server(repo, queue, provider)

    # Semantic + FTS5 hybrid search
    res = await server.call_tool("search_photos", {"query": "ocean sunset"})
    items = _extract_result(res)
    assert len(items) >= 1
    assert items[0]["id"] == p1
    assert provider.calls[0] == ["ocean sunset"]

    # Date range + tag filter
    res_filtered = await server.call_tool(
        "search_photos",
        {
            "query": "mountain",
            "start_date": "2024-08-01T00:00:00Z",
            "end_date": "2024-08-31T23:59:59Z",
            "tags": ["hike"],
        },
    )
    filtered_items = _extract_result(res_filtered)
    assert [item["id"] for item in filtered_items] == [p2]

    # Year-only SQL filter without query
    res_2023 = await server.call_tool("search_photos", {"year": 2023})
    items_2023 = _extract_result(res_2023)
    assert [item["id"] for item in items_2023] == [p3]


async def test_search_photos_fts_fallback_on_embed_error(db_env) -> None:
    _, repo, queue = db_env
    p1, _, _ = _seed_photos(repo)
    failing_provider = MockEmbedProvider(fail=True)
    server = create_mcp_server(repo, queue, failing_provider)

    res = await server.call_tool("search_photos", {"query": "ocean"})
    items = _extract_result(res)
    assert len(items) == 1
    assert items[0]["id"] == p1
    assert failing_provider.calls == [["ocean"]]

    # Also works when embed_provider is None
    server_no_embed = create_mcp_server(repo, queue, None)
    res_no_embed = await server_no_embed.call_tool("search_photos", {"query": "ocean"})
    items_no_embed = _extract_result(res_no_embed)
    assert [item["id"] for item in items_no_embed] == [p1]


async def test_photo_details_and_year_stats(db_env) -> None:
    _, repo, queue = db_env
    p1, p2, _ = _seed_photos(repo)
    server = create_mcp_server(repo, queue, MockEmbedProvider())

    details = _extract_result(await server.call_tool("get_photo_details", {"photo_id": p1}))
    assert details is not None
    assert details["picker_id"] == "pick-1"
    assert set(details["tags"]) == {"beach", "ocean", "sunset"}

    missing = _extract_result(await server.call_tool("get_photo_details", {"photo_id": 99999}))
    assert missing is None

    stats = _extract_result(await server.call_tool("get_year_stats", {"year": 2024}))
    assert stats["year"] == 2024
    assert stats["total_photos"] == 2
    assert stats["indexed_photos"] == 2
    assert len(stats["by_month"]) == 2
    assert any(t["tag"] == "beach" for t in stats["top_tags"])


async def test_queue_status_control_and_reprocess(db_env) -> None:
    _, repo, queue = db_env
    p1, p2, _ = _seed_photos(repo)
    j1 = queue.enqueue("download", p1)
    j2 = queue.enqueue("vision", p2, max_attempts=1)
    claimed = queue.claim_next("worker-1", job_types=["vision"])
    assert claimed is not None and claimed["id"] == j2
    queue.fail(j2, "boom")

    server = create_mcp_server(repo, queue, MockEmbedProvider())

    status = _extract_result(await server.call_tool("get_queue_status", {}))
    assert status["total"] == 2
    assert status["by_status"]["pending"] == 1
    assert status["by_status"]["failed"] == 1

    # Pause pending download job
    paused = _extract_result(
        await server.call_tool("control_queue", {"action": "pause", "job_type": "download"})
    )
    assert paused["affected"] == 1
    assert paused["queue"]["by_status"]["paused"] == 1

    # Resume paused download job
    resumed = _extract_result(
        await server.call_tool("control_queue", {"action": "resume", "job_type": "download"})
    )
    assert resumed["affected"] == 1
    assert resumed["queue"]["by_status"]["pending"] == 1

    # Retry failed vision job
    retried = _extract_result(
        await server.call_tool("control_queue", {"action": "retry_failed"})
    )
    assert retried["affected"] == 1
    assert retried["queue"]["by_status"]["pending"] == 2

    # Reprocess photo enqueues a new vision job
    reproc = _extract_result(await server.call_tool("reprocess_photo", {"photo_id": p1}))
    assert reproc["photo_id"] == p1
    assert reproc["job_type"] == "vision"
    assert reproc["status"] == "pending"
    assert reproc["job_id"] > j1


async def test_group_crud_reorder_and_repick_manifest(db_env) -> None:
    _, repo, queue = db_env
    p1, p2, p3 = _seed_photos(repo)
    server = create_mcp_server(repo, queue, MockEmbedProvider())

    g1 = _extract_result(
        await server.call_tool(
            "create_group",
            {"name": "Summer 2024", "year": 2024, "description": "Best summer shots"},
        )
    )
    assert g1["name"] == "Summer 2024"
    assert g1["year"] == 2024

    await server.call_tool(
        "create_group",
        {"name": "Winter 2023", "year": 2023, "kind": "book_chapter"},
    )

    groups_2024 = _extract_result(await server.call_tool("list_groups", {"year": 2024}))
    assert len(groups_2024) == 1
    assert groups_2024[0]["id"] == g1["id"]

    all_groups = _extract_result(await server.call_tool("list_groups", {}))
    assert len(all_groups) == 2

    # Add photos with string-keyed captions
    add_res = _extract_result(
        await server.call_tool(
            "add_photos_to_group",
            {
                "group_id": g1["id"],
                "photo_ids": [p1, p2, p3],
                "captions": {str(p1): "Sunset by the sea", str(p2): "Up the ridge"},
            },
        )
    )
    assert add_res["added"] == 3

    # Reorder group
    await server.call_tool(
        "reorder_group",
        {"group_id": g1["id"], "ordered_photo_ids": [p2, p1, p3]},
    )

    # Remove p3
    rem_res = _extract_result(
        await server.call_tool(
            "remove_photos_from_group",
            {"group_id": g1["id"], "photo_ids": [p3]},
        )
    )
    assert rem_res["removed"] == 1

    group_photos = _extract_result(
        await server.call_tool("get_group_photos", {"group_id": g1["id"]})
    )
    assert [p["id"] for p in group_photos] == [p2, p1]
    assert group_photos[0]["caption"] == "Up the ridge"
    assert group_photos[1]["caption"] == "Sunset by the sea"

    # Search filtered by group_id
    group_search = _extract_result(
        await server.call_tool("search_photos", {"group_id": g1["id"]})
    )
    assert {p["id"] for p in group_search} == {p1, p2}

    # Export repick manifest
    repo.mark_original_downloaded(p1, "data/originals/2024/1.jpg")
    manifest = _extract_result(
        await server.call_tool("export_repick_manifest", {"group_id": g1["id"]})
    )
    assert [m["photo_id"] for m in manifest] == [p2, p1]
    assert manifest[0]["needs_original"] is True
    assert manifest[1]["needs_original"] is False


async def test_save_book_draft(db_env) -> None:
    _, repo, queue = db_env
    p1, p2, _ = _seed_photos(repo)
    server = create_mcp_server(repo, queue, MockEmbedProvider())

    chapters = [
        {"title": "Summer Escapes", "photo_ids": [p1, p2], "narrative": "Warm days."}
    ]
    book = _extract_result(
        await server.call_tool(
            "save_book_draft",
            {"year": 2024, "title": "Our 2024 Memory Book", "chapters_json": chapters},
        )
    )
    assert book["year"] == 2024
    assert book["title"] == "Our 2024 Memory Book"
    parsed_outline = json.loads(book["outline_json"])
    assert parsed_outline == {"chapters": chapters}
