"""Tests for the FastAPI local web UI and export endpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from vision_memory_book.config import Settings
from vision_memory_book.db import JobQueue, Repository, init_db
from vision_memory_book.web import create_app


def _write_test_image(path: Path, color: tuple[int, int, int], fmt: str = "WEBP") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path, format=fmt)


@pytest.fixture
def web_env(tmp_path: Path):
    db_path = tmp_path / "vmb_web.db"
    cache_dir = tmp_path / "cache"
    originals_dir = tmp_path / "originals"

    settings = Settings(
        data_dir=tmp_path,
        db_path=db_path,
        cache_dir=cache_dir,
        originals_dir=originals_dir,
        embed_dim=4,
    )
    conn = init_db(db_path, embed_dim=4)
    repo = Repository(conn)
    queue = JobQueue(conn)

    # Create cached WebP files and one original JPEG
    p1_cache = cache_dir / "2024" / "1.webp"
    p2_cache = cache_dir / "2024" / "2.webp"
    p3_cache = cache_dir / "2023" / "3.webp"
    p2_orig = originals_dir / "2024" / "2_orig.jpg"

    _write_test_image(p1_cache, (255, 0, 0), fmt="WEBP")
    _write_test_image(p2_cache, (0, 255, 0), fmt="WEBP")
    _write_test_image(p3_cache, (0, 0, 255), fmt="WEBP")
    _write_test_image(p2_orig, (0, 200, 0), fmt="JPEG")

    p1 = repo.upsert_photo(
        picker_id="pick-1",
        filename="sunset_beach.jpg",
        create_time="2024-07-15T18:30:00Z",
        width=1024,
        height=768,
    )
    repo.mark_downloaded(p1, str(p1_cache), 1024, 768)
    repo.save_vision_index(
        photo_id=p1,
        description="Golden sunset over the sandy beach",
        tags=["beach", "sunset", "summer"],
        vlm_json={"scene": "beach"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )

    p2 = repo.upsert_photo(
        picker_id="pick-2",
        filename="mountain_hike.jpg",
        create_time="2024-08-20T11:00:00Z",
        width=1024,
        height=768,
    )
    repo.mark_downloaded(p2, str(p2_cache), 1024, 768)
    repo.mark_original_downloaded(p2, str(p2_orig))
    repo.save_vision_index(
        photo_id=p2,
        description="Alpine trail with pine trees and mountain peaks",
        tags=["mountain", "hiking"],
        vlm_json={"scene": "mountain"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )

    p3 = repo.upsert_photo(
        picker_id="pick-3",
        filename="winter_cabin.jpg",
        create_time="2023-12-25T09:00:00Z",
        width=1024,
        height=768,
    )
    repo.mark_downloaded(p3, str(p3_cache), 1024, 768)
    repo.save_vision_index(
        photo_id=p3,
        description="Cozy snowy cabin in winter",
        tags=["winter", "snow"],
        vlm_json={"scene": "winter"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[0.0, 0.0, 1.0, 0.0],
    )

    app = create_app(settings=settings, repo=repo, queue=queue)
    client = TestClient(app)

    try:
        yield {
            "client": client,
            "repo": repo,
            "queue": queue,
            "photos": [p1, p2, p3],
            "p1_cache": p1_cache,
            "p2_cache": p2_cache,
            "p2_orig": p2_orig,
        }
    finally:
        conn.close()


def test_dashboard_and_queue_control(web_env) -> None:
    client: TestClient = web_env["client"]
    queue: JobQueue = web_env["queue"]
    p1, p2, p3 = web_env["photos"]

    j_pending = queue.enqueue("download", p1, payload={"base_url": "https://example.com/1"})
    j_fail = queue.enqueue("vision", p2, max_attempts=1)
    claimed = queue.claim_next("worker-test", job_types=["vision"])
    assert claimed is not None and claimed["id"] == j_fail
    queue.fail(j_fail, "timeout")

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text
    assert "2024" in resp.text
    assert "2023" in resp.text
    assert "timeout" in resp.text

    # Pause pending jobs via form
    resp = client.post("/queue/control", data={"action": "pause"})
    assert resp.status_code == 200
    stats = queue.stats()
    assert stats["by_status"].get("paused", 0) == 1
    assert stats["by_status"].get("pending", 0) == 0

    # Resume paused jobs via JSON
    resp = client.post("/queue/control", json={"action": "resume"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "resume"
    assert data["affected"] == 1
    assert queue.stats()["by_status"].get("pending", 0) == 1

    # Retry failed jobs
    resp = client.post("/queue/control", json={"action": "retry_failed"})
    assert resp.status_code == 200
    assert resp.json()["affected"] == 1
    assert queue.stats()["by_status"].get("failed", 0) == 0
    assert queue.stats()["by_status"].get("pending", 0) == 2

    # Invalid action returns 400
    bad_resp = client.post("/queue/control", json={"action": "nope"})
    assert bad_resp.status_code == 400


def test_gallery_search_and_filters(web_env) -> None:
    client: TestClient = web_env["client"]

    # FTS5 query
    resp = client.get("/gallery", params={"q": "sunset"})
    assert resp.status_code == 200
    assert "sunset_beach.jpg" in resp.text
    assert "mountain_hike.jpg" not in resp.text

    # Year filter
    resp = client.get("/gallery", params={"year": "2023"})
    assert resp.status_code == 200
    assert "winter_cabin.jpg" in resp.text
    assert "sunset_beach.jpg" not in resp.text

    # Tag filter
    resp = client.get("/gallery", params={"tag": "hiking"})
    assert resp.status_code == 200
    assert "mountain_hike.jpg" in resp.text
    assert "winter_cabin.jpg" not in resp.text

    # Date range filter
    resp = client.get(
        "/gallery",
        params={"start_date": "2024-08-01T00:00:00Z", "end_date": "2024-09-01T00:00:00Z"},
    )
    assert resp.status_code == 200
    assert "mountain_hike.jpg" in resp.text
    assert "sunset_beach.jpg" not in resp.text


def test_photo_image_serving(web_env) -> None:
    client: TestClient = web_env["client"]
    p1, p2, _ = web_env["photos"]
    p1_cache: Path = web_env["p1_cache"]
    p2_cache: Path = web_env["p2_cache"]
    p2_orig: Path = web_env["p2_orig"]

    # Default serves cached WebP
    r1 = client.get(f"/photos/{p1}/image")
    assert r1.status_code == 200
    assert r1.content == p1_cache.read_bytes()

    # p1 has no original_path, so prefer_original=true falls back to cache_path
    r1_pref = client.get(f"/photos/{p1}/image", params={"prefer_original": "true"})
    assert r1_pref.status_code == 200
    assert r1_pref.content == p1_cache.read_bytes()

    # p2 has original_path, so default serves cache, prefer_original=true serves original
    r2_cache = client.get(f"/photos/{p2}/image")
    assert r2_cache.status_code == 200
    assert r2_cache.content == p2_cache.read_bytes()

    r2_orig = client.get(f"/photos/{p2}/image", params={"prefer_original": "true"})
    assert r2_orig.status_code == 200
    assert r2_orig.content == p2_orig.read_bytes()

    # Missing photo returns 404
    assert client.get("/photos/99999/image").status_code == 404


def test_groups_crud_and_zip_download(web_env) -> None:
    client: TestClient = web_env["client"]
    repo: Repository = web_env["repo"]
    p1, p2, p3 = web_env["photos"]
    p1_cache: Path = web_env["p1_cache"]
    p2_orig: Path = web_env["p2_orig"]

    # Create group via form
    resp = client.post(
        "/groups",
        data={"name": "Best of 2024", "year": "2024", "description": "Highlights"},
    )
    assert resp.status_code == 200
    assert "Best of 2024" in resp.text

    groups = repo.list_groups()
    assert len(groups) == 1
    group_id = groups[0]["id"]

    # Add p1 and p2 via JSON
    resp = client.post(
        f"/groups/{group_id}/items",
        json={
            "action": "add",
            "photo_ids": [p1, p2],
            "captions": {str(p1): "Beach evening", str(p2): "Summit view"},
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["photos"]) == 2

    # Add p3 and then remove p3 via form
    client.post(f"/groups/{group_id}/items", data={"action": "add", "photo_ids": str(p3)})
    assert len(repo.get_group_photos(group_id)) == 3

    client.post(f"/groups/{group_id}/items", data={"action": "remove", "photo_id": str(p3)})
    assert [p["id"] for p in repo.get_group_photos(group_id)] == [p1, p2]

    # Reorder [p2, p1]
    client.post(
        f"/groups/{group_id}/items",
        data={"action": "reorder", "ordered_photo_ids": f"{p2},{p1}"},
    )
    assert [p["id"] for p in repo.get_group_photos(group_id)] == [p2, p1]

    # Download ZIP and verify contents (p2 uses original, p1 uses cache, metadata.json present)
    zip_resp = client.get(f"/groups/{group_id}/download.zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(zip_resp.content), "r") as zf:
        names = set(zf.namelist())
        assert "metadata.json" in names
        assert "001_mountain_hike.jpg" in names
        assert "002_sunset_beach.webp" in names

        assert zf.read("001_mountain_hike.jpg") == p2_orig.read_bytes()
        assert zf.read("002_sunset_beach.webp") == p1_cache.read_bytes()

        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        assert meta["group"]["name"] == "Best of 2024"
        assert len(meta["items"]) == 2
        assert meta["items"][0]["caption"] == "Summit view"
        assert meta["items"][0]["used_original"] is True
        assert meta["items"][1]["caption"] == "Beach evening"
        assert meta["items"][1]["used_original"] is False


def test_repick_workflow_manifest_and_match(web_env) -> None:
    client: TestClient = web_env["client"]
    repo: Repository = web_env["repo"]
    queue: JobQueue = web_env["queue"]
    p1, p2, _ = web_env["photos"]

    group_id = repo.create_group("Re-Pick Test Group", year=2024)
    repo.add_photos_to_group(group_id, [p1, p2])

    # HTML page lists pick-1 (missing original) in checklist
    page_resp = client.get(f"/groups/{group_id}/repick")
    assert page_resp.status_code == 200
    assert "pick-1" in page_resp.text
    assert "sunset_beach.jpg" in page_resp.text
    assert "Already Has Original (1)" in page_resp.text

    # Manifest JSON
    manifest_resp = client.get(f"/groups/{group_id}/repick/manifest.json")
    assert manifest_resp.status_code == 200
    manifest_data = manifest_resp.json()
    assert manifest_data["group_id"] == group_id
    assert manifest_data["total_count"] == 2
    assert manifest_data["missing_count"] == 1
    by_pid = {item["picker_id"]: item for item in manifest_data["items"]}
    assert by_pid["pick-1"]["needs_original"] is True
    assert by_pid["pick-2"]["needs_original"] is False

    # Match newly picked items -> enqueues repick_original job
    match_resp = client.post(
        f"/groups/{group_id}/repick/match",
        json=[
            {"picker_id": "pick-1", "base_url": "https://lh3.googleusercontent.com/new-1"},
            {"picker_id": "pick-unknown", "base_url": "https://lh3.googleusercontent.com/other"},
        ],
    )
    assert match_resp.status_code == 200
    match_data = match_resp.json()
    assert match_data["matched_count"] == 1
    assert match_data["matched"][0]["photo_id"] == p1
    assert match_data["matched"][0]["picker_id"] == "pick-1"
    assert match_data["unmatched"] == ["pick-unknown"]

    # Verify job is in JobQueue as repick_original
    claimed = queue.claim_next("repick-worker", job_types=["repick_original"])
    assert claimed is not None
    assert claimed["photo_id"] == p1
    assert claimed["payload"]["base_url"] == "https://lh3.googleusercontent.com/new-1"
    assert claimed["payload"]["group_id"] == group_id
