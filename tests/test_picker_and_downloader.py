"""Tests for Google Photos Picker client, OAuth manager, and SSD-safe DownloadWorker."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image

from vision_memory_book.config import Settings
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.picker.auth import GoogleOAuthManager
from vision_memory_book.picker.client import PickerClient, ingest_picker_session
from vision_memory_book.workers.downloader import DownloadWorker


def _make_jpeg_bytes(width: int = 120, height: int = 80, orientation: int | None = None) -> bytes:
    """Create valid JPEG bytes in memory, optionally with an EXIF orientation tag."""
    img = Image.new("RGB", (width, height), color=(42, 128, 200))
    buf = io.BytesIO()
    if orientation is not None:
        exif = img.getexif()
        exif[0x0112] = orientation  # Orientation tag (6 = 90 deg CW)
        img.save(buf, format="JPEG", exif=exif.tobytes())
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


async def test_oauth_manager_persistence_and_auto_refresh(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://oauth2.googleapis.com/token"
        body = request.content.decode("utf-8")
        calls.append({"body": body})
        return httpx.Response(
            200,
            json={
                "access_token": "new-access-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        google_client_id="cid-123",
        google_client_secret="csec-456",
        google_token_path=token_path,
    )
    mgr = GoogleOAuthManager(settings=settings, http_client=async_client)

    # Save an expired token with a refresh_token
    mgr.save_token(
        {
            "access_token": "old-expired-token",
            "refresh_token": "refresh-xyz",
            "expires_at": time.time() - 10,
        }
    )

    token = await mgr.get_access_token()
    assert token == "new-access-token"
    assert len(calls) == 1
    assert "grant_type=refresh_token" in calls[0]["body"]

    # Verify persisted token retained refresh_token and updated access_token
    persisted = json.loads(token_path.read_text(encoding="utf-8"))
    assert persisted["access_token"] == "new-access-token"
    assert persisted["refresh_token"] == "refresh-xyz"
    await async_client.aclose()


async def test_picker_client_sessions_pagination_and_download_urls() -> None:
    seen_requests: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(
            (request.method, str(request.url), request.headers.get("Authorization"))
        )
        url = str(request.url)
        if request.method == "POST" and url.endswith("/v1/sessions"):
            return httpx.Response(
                200,
                json={"id": "sess-1", "pickerUri": "https://photos.google.com/picker/sess-1"},
            )
        if request.method == "GET" and url.endswith("/v1/sessions/sess-1"):
            return httpx.Response(
                200,
                json={"id": "sess-1", "mediaItemsSet": True},
            )
        if request.method == "DELETE" and url.endswith("/v1/sessions/sess-1"):
            return httpx.Response(200, json={})
        if request.method == "GET" and "/v1/mediaItems" in url:
            if "pageToken=page-2" in url:
                return httpx.Response(
                    200,
                    json={
                        "mediaItems": [
                            {"id": "p3", "mediaFile": {"baseUrl": "https://lh3.test/p3"}}
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "mediaItems": [
                        {"id": "p1", "mediaFile": {"baseUrl": "https://lh3.test/p1"}},
                        {"id": "p2", "mediaFile": {"baseUrl": "https://lh3.test/p2"}},
                    ],
                    "nextPageToken": "page-2",
                },
            )
        if url == "https://lh3.test/p1=w1024-h1024":
            return httpx.Response(200, content=b"resized-bytes")
        if url == "https://lh3.test/p1=d":
            return httpx.Response(200, content=b"original-bytes")
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with PickerClient(http_client=http_client, access_token="tok-abc") as client:
        created = await client.create_session()
        assert created["id"] == "sess-1"

        fetched = await client.get_session("sess-1")
        assert fetched["mediaItemsSet"] is True

        items = await client.list_media_items("sess-1", page_size=2)
        assert [item["id"] for item in items] == ["p1", "p2", "p3"]

        resized = await client.download_media_bytes("https://lh3.test/p1", max_dim=1024)
        assert resized == b"resized-bytes"

        original = await client.download_media_bytes("https://lh3.test/p1", original=True)
        assert original == b"original-bytes"

        await client.delete_session("sess-1")

    await http_client.aclose()
    assert all(auth == "Bearer tok-abc" for _, _, auth in seen_requests)


async def test_ingest_picker_session_dedup_and_enqueue(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    # Pre-existing indexed photo should not be re-enqueued
    existing_id = repo.upsert_photo(picker_id="pick-indexed", filename="old.jpg")
    repo.save_vision_index(
        photo_id=existing_id,
        description="Already indexed",
        tags=["summer"],
        vlm_json={"description": "Already indexed"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[0.1] * 768,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mediaItems": [
                    {
                        "id": "pick-indexed",
                        "createTime": "2024-06-01T10:00:00Z",
                        "mediaFile": {
                            "baseUrl": "https://lh3.test/old",
                            "mimeType": "image/jpeg",
                            "filename": "old.jpg",
                        },
                    },
                    {
                        "id": "pick-new",
                        "createTime": "2024-07-15T12:00:00Z",
                        "mediaFile": {
                            "baseUrl": "https://lh3.test/new",
                            "mimeType": "image/jpeg",
                            "filename": "new.jpg",
                            "mediaFileMetadata": {
                                "width": 4000,
                                "height": 3000,
                                "cameraMake": "Google",
                                "cameraModel": "Pixel 8",
                            },
                        },
                    },
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = PickerClient(http_client=http_client, access_token="tok-1")
    summary = await ingest_picker_session(client, repo, queue, "sess-ingest")

    assert summary == {"total": 2, "enqueued": 1, "skipped": 1}

    new_photo = repo.get_photo_by_picker_id("pick-new")
    assert new_photo is not None
    assert new_photo["year"] == 2024
    assert new_photo["camera_model"] == "Pixel 8"

    job = queue.claim_next("test-worker", job_types=["download"])
    assert job is not None
    assert job["photo_id"] == new_photo["id"]
    assert job["payload"]["base_url"] == "https://lh3.test/new"
    assert queue.claim_next("test-worker", job_types=["download"]) is None

    await http_client.aclose()
    conn.close()


async def test_download_worker_webp_and_repick_original(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    settings = Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "vmb.db",
        cache_dir=tmp_path / "cache",
        originals_dir=tmp_path / "originals",
        cache_max_dim=1024,
    )
    settings.ensure_dirs()

    # Create JPEG with EXIF orientation=6 (90deg CW) so 120x80 becomes 80x120 after transpose
    resized_jpeg = _make_jpeg_bytes(width=120, height=80, orientation=6)
    original_jpeg = _make_jpeg_bytes(width=400, height=300)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://lh3.test/item1=w1024-h1024":
            return httpx.Response(200, content=resized_jpeg)
        if url == "https://lh3.test/item1=d":
            return httpx.Response(200, content=original_jpeg)
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    picker_client = PickerClient(http_client=http_client, access_token="tok-dl")
    worker = DownloadWorker(
        repo=repo,
        queue=queue,
        picker_client=picker_client,
        settings=settings,
    )

    photo_id = repo.upsert_photo(
        picker_id="item1",
        filename="vacation/IMG 001.jpg",
        mime_type="image/jpeg",
        create_time="2023-08-10T15:30:00Z",
    )
    queue.enqueue("download", photo_id, payload={"base_url": "https://lh3.test/item1"})

    # 1. Process download job -> WebP cache + vision job enqueued
    processed = await worker.run_once("dl-worker-1")
    assert processed is True

    expected_cache = settings.cache_dir / "2023" / f"{photo_id}.webp"
    assert expected_cache.exists()

    with Image.open(expected_cache) as cached_img:
        assert cached_img.format == "WEBP"
        assert cached_img.size == (80, 120)

    photo = repo.get_photo(photo_id)
    assert photo is not None
    assert photo["status"] == "downloaded"
    assert photo["cache_path"] == str(expected_cache)
    assert photo["width"] == 80
    assert photo["height"] == 120

    vision_job = queue.claim_next("vision-worker-1", job_types=["vision"])
    assert vision_job is not None
    assert vision_job["photo_id"] == photo_id

    # 2. Process repick_original job -> originals_dir
    queue.enqueue(
        "repick_original",
        photo_id,
        payload={"base_url": "https://lh3.test/item1", "filename": "vacation/IMG 001.jpg"},
    )
    processed_orig = await worker.run_once("dl-worker-1")
    assert processed_orig is True

    expected_orig = settings.originals_dir / f"{photo_id}_IMG_001.jpg"
    assert expected_orig.exists()
    assert expected_orig.read_bytes() == original_jpeg

    photo_after = repo.get_photo(photo_id)
    assert photo_after is not None
    assert photo_after["original_path"] == str(expected_orig)

    # No more jobs
    assert await worker.run_once("dl-worker-1") is False

    await http_client.aclose()
    conn.close()


async def test_download_worker_handles_expired_baseurl_403(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "vmb.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    settings = Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "vmb.db",
        cache_dir=tmp_path / "cache",
        originals_dir=tmp_path / "originals",
    )
    settings.ensure_dirs()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="URL expired")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    picker_client = PickerClient(http_client=http_client, access_token="tok-exp")
    worker = DownloadWorker(
        repo=repo,
        queue=queue,
        picker_client=picker_client,
        settings=settings,
    )

    photo_id = repo.upsert_photo(picker_id="expired-1", filename="exp.jpg")
    job_id = queue.enqueue(
        "download",
        photo_id,
        payload={"base_url": "https://lh3.test/expired"},
        max_attempts=1,
    )

    processed = await worker.run_once("dl-worker-err")
    assert processed is True

    row = conn.execute("SELECT status, last_error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "failed"
    assert "403" in row["last_error"]

    photo = repo.get_photo(photo_id)
    assert photo is not None
    assert photo["status"] == "failed"

    # Also verify run_forever exits cleanly when stop_event is set
    stop_event = asyncio.Event()
    stop_event.set()
    await worker.run_forever("dl-worker-err", poll_interval=0.05, stop_event=stop_event)

    await http_client.aclose()
    conn.close()
