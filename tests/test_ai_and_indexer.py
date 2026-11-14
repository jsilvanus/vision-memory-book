"""Tests for Aidos discovery, OpenAICompatProvider, and VisionIndexerWorker."""

import asyncio
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from vision_memory_book.ai.aidos_discovery import AidosEndpoint, discover_aidos_endpoint
from vision_memory_book.ai.base import AIProvider
from vision_memory_book.ai.openai_compat import OpenAICompatProvider, build_ai_providers
from vision_memory_book.config import Settings
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.workers.indexer import VisionIndexerWorker


def test_discover_aidos_endpoint_with_base_url_and_token(tmp_path: Path) -> None:
    disc_file = tmp_path / "aidos.json"
    disc_file.write_text(
        json.dumps({"base_url": "http://127.0.0.1:19001", "token": "secret-tok"}),
        encoding="utf-8",
    )

    ep = discover_aidos_endpoint(disc_file)
    assert ep == AidosEndpoint(base_url="http://127.0.0.1:19001/v1", token="secret-tok")


def test_discover_aidos_endpoint_with_port_and_bearer_token(tmp_path: Path) -> None:
    disc_file = tmp_path / "aidos.json"
    disc_file.write_text(
        json.dumps({"port": 24567, "bearer_token": "loopback-abc"}),
        encoding="utf-8",
    )

    ep = discover_aidos_endpoint(disc_file)
    assert ep == AidosEndpoint(base_url="http://127.0.0.1:24567/v1", token="loopback-abc")


def test_discover_aidos_endpoint_fallback_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.json"
    ep = discover_aidos_endpoint(missing, default_port=18080)
    assert ep == AidosEndpoint(base_url="http://127.0.0.1:18080/v1", token=None)

    ep_none = discover_aidos_endpoint(None, default_port=18123)
    assert ep_none == AidosEndpoint(base_url="http://127.0.0.1:18123/v1", token=None)


@pytest.mark.asyncio
async def test_openai_compat_describe_image_json_and_fenced_and_fallback() -> None:
    captured_headers: list[str | None] = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "description": "Golden sunset over calm ocean waves.",
                                "tags": ["Sunset", "Ocean", "Beach"],
                                "scene": "beach sunset",
                                "subjects": ["ocean", "sun"],
                            }
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"description": "A cozy family dinner.", '
                            '"tags": ["Family", "Dinner"], '
                            '"scene": "dining room", '
                            '"subjects": ["table", "people"]}\n'
                            "```"
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "Plain text description without JSON formatting."
                    }
                }
            ]
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(request.headers.get("authorization"))
        body = json.loads(request.content.decode("utf-8"))
        img_block = body["messages"][0]["content"][1]
        assert img_block["type"] == "image_url"
        assert img_block["image_url"]["url"].startswith("data:image/webp;base64,")
        return httpx.Response(200, json=responses.pop(0))

    provider = OpenAICompatProvider(
        base_url="http://127.0.0.1:18080/v1",
        api_key="tok-123",
        vision_model="qwen2.5vl",
        transport=httpx.MockTransport(handler),
    )
    assert isinstance(provider, AIProvider)

    try:
        res1 = await provider.describe_image(b"fake-webp-bytes", "image/webp")
        assert res1.description == "Golden sunset over calm ocean waves."
        assert res1.tags == ["sunset", "ocean", "beach"]
        assert res1.scene == "beach sunset"
        assert res1.subjects == ["ocean", "sun"]
        assert res1.raw_json["scene"] == "beach sunset"

        res2 = await provider.describe_image(b"fake-webp-bytes", "image/webp")
        assert res2.description == "A cozy family dinner."
        assert res2.tags == ["family", "dinner"]
        assert res2.scene == "dining room"

        res3 = await provider.describe_image(b"fake-webp-bytes", "image/webp")
        assert res3.description == "Plain text description without JSON formatting."
        assert res3.tags == []
        assert res3.raw_json == {}

        assert captured_headers == ["Bearer tok-123"] * 3
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_openai_compat_embed_and_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer embed-key"
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    ]
                },
            )
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Chapter 1: Summer"}}]},
            )
        return httpx.Response(404)

    provider = OpenAICompatProvider(
        base_url="http://localhost:11434/v1",
        api_key="embed-key",
        embed_model="nomic-embed-text",
        text_model="qwen2.5:14b",
        transport=httpx.MockTransport(handler),
    )
    try:
        vectors = await provider.embed(["first", "second"])
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        reply = await provider.chat([{"role": "user", "content": "Write a chapter title."}])
        assert reply == "Chapter 1: Summer"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_build_ai_providers_split_config_with_aidos_discovery(tmp_path: Path) -> None:
    disc_file = tmp_path / "aidos_discovery.json"
    disc_file.write_text(
        json.dumps({"port": 19999, "token": "aidos-loopback-token"}),
        encoding="utf-8",
    )
    settings = Settings(
        vision_provider="ollama",
        vision_base_url="http://localhost:11434/v1",
        vision_model="qwen2.5vl",
        embed_provider="aidos",
        embed_model="nomic-embed-text",
        text_provider="aidos",
        text_model="qwen2.5:14b",
        aidos_discovery_path=str(disc_file),
    )

    providers = build_ai_providers(settings)
    try:
        assert set(providers.keys()) == {"vision", "embed", "text"}
        assert providers["vision"].base_url == "http://localhost:11434/v1"
        assert providers["vision"].api_key is None
        assert providers["embed"].base_url == "http://127.0.0.1:19999/v1"
        assert providers["embed"].api_key == "aidos-loopback-token"
        assert providers["text"].base_url == "http://127.0.0.1:19999/v1"
        assert providers["text"].api_key == "aidos-loopback-token"
    finally:
        for p in providers.values():
            await p.aclose()


@pytest.mark.asyncio
async def test_vision_indexer_worker_run_once_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "test_vmb.db"
    conn = init_db(db_path, embed_dim=4)
    repo = Repository(conn)
    queue = JobQueue(conn)

    # Create a real tiny WebP image on disk
    webp_path = tmp_path / "sample.webp"
    img = Image.new("RGB", (16, 16), color=(255, 128, 0))
    img.save(webp_path, format="WEBP")

    photo_id = repo.upsert_photo(
        picker_id="picker-123",
        filename="sunset.webp",
        mime_type="image/webp",
        create_time="2025-07-15T18:30:00Z",
        width=16,
        height=16,
    )
    repo.mark_downloaded(photo_id, cache_path=str(webp_path), width=16, height=16)
    job_id = queue.enqueue("vision", photo_id)

    seen_embed_inputs: list[str] = []

    def ai_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "description": "Warm orange sunset over alpine lake.",
                                        "tags": ["Sunset", "Alpine", "Lake"],
                                        "scene": "mountain lake",
                                        "subjects": ["lake", "mountains"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        if request.url.path.endswith("/embeddings"):
            seen_embed_inputs.extend(body["input"])
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.9, 0.1, 0.0, 0.2]}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(ai_handler)
    vision_provider = OpenAICompatProvider(
        base_url="http://localhost:11434/v1",
        vision_model="qwen2.5vl",
        transport=transport,
    )
    embed_provider = OpenAICompatProvider(
        base_url="http://127.0.0.1:18080/v1",
        embed_model="nomic-embed-text",
        transport=transport,
    )

    try:
        worker = VisionIndexerWorker(
            queue=queue,
            repo=repo,
            vision_provider=vision_provider,
            embed_provider=embed_provider,
        )

        processed = await worker.run_once(worker_id="worker-test")
        assert processed is True

        # Empty queue returns False
        processed_again = await worker.run_once(worker_id="worker-test")
        assert processed_again is False

        # Verify embedding input format
        assert seen_embed_inputs == [
            "Warm orange sunset over alpine lake.\nTags: sunset, alpine, lake"
        ]

        # Verify job completed
        job_row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert job_row["status"] == "done"

        # Verify photos & photo_tags populated
        photo = repo.get_photo(photo_id)
        assert photo is not None
        assert photo["status"] == "indexed"
        assert photo["description"] == "Warm orange sunset over alpine lake."
        assert photo["vlm_model"] == "qwen2.5vl"
        assert photo["embed_model"] == "nomic-embed-text"
        assert photo["tags"] == ["alpine", "lake", "sunset"]

        # Verify FTS5 search works
        fts_hits = repo.search_photos(text_query="alpine")
        assert len(fts_hits) == 1
        assert fts_hits[0]["id"] == photo_id

        # Verify vector search works
        vec_hits = repo.search_photos(query_embedding=[0.9, 0.1, 0.0, 0.2])
        assert len(vec_hits) == 1
        assert vec_hits[0]["id"] == photo_id

        # Verify stop_event cleanly stops run_forever
        stop_event = asyncio.Event()
        stop_event.set()
        await worker.run_forever(worker_id="worker-test", poll_interval=0.05, stop_event=stop_event)
    finally:
        await vision_provider.aclose()
        await embed_provider.aclose()
        conn.close()
