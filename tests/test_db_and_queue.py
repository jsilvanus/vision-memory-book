"""Tests for database initialization, JobQueue, and Repository."""

from pathlib import Path
import sqlite3

import pytest

from vision_memory_book.config import Settings, get_settings
from vision_memory_book.db import JobQueue, Repository, init_db


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    db_file = tmp_path / "test_vmb.db"
    conn = init_db(db_file, embed_dim=4)
    yield conn
    conn.close()


def test_init_db_wal_mode_and_tables(tmp_path: Path) -> None:
    db_file = tmp_path / "wal_check.db"
    conn = init_db(db_file, embed_dim=4)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        fk_enabled = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk_enabled == 1

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        expected = {
            "picker_sessions",
            "photos",
            "photo_tags",
            "photos_fts",
            "photo_embeddings",
            "jobs",
            "groups",
            "group_items",
            "books",
        }
        assert expected.issubset(tables)
    finally:
        conn.close()


def test_settings_defaults() -> None:
    get_settings.cache_clear()
    settings = Settings()
    assert settings.embed_dim == 768
    assert settings.cache_max_dim == 1024
    assert settings.vision_base_url == "http://localhost:11434/v1"


def test_job_queue_lifecycle_and_priority(db_conn: sqlite3.Connection) -> None:
    repo = Repository(db_conn)
    queue = JobQueue(db_conn)

    p1 = repo.upsert_photo(picker_id="pick-1", filename="a.jpg", create_time="2024-06-01T10:00:00Z")
    p2 = repo.upsert_photo(picker_id="pick-2", filename="b.jpg", create_time="2024-06-02T10:00:00Z")

    j1 = queue.enqueue("download", p1, payload={"base_url": "https://example.com/1"}, priority=1)
    j2 = queue.enqueue("download", p2, payload={"base_url": "https://example.com/2"}, priority=10)
    j3 = queue.enqueue("vision", p1, priority=5)

    # Higher priority download job claimed first
    claimed = queue.claim_next("worker-1", job_types=["download"])
    assert claimed is not None
    assert claimed["id"] == j2
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    assert claimed["payload"]["base_url"] == "https://example.com/2"

    assert queue.heartbeat(j2, worker_id="worker-1") is True
    assert queue.complete(j2) is True

    # Next download job
    claimed2 = queue.claim_next("worker-1", job_types=["download"])
    assert claimed2 is not None
    assert claimed2["id"] == j1

    # Vision job still pending
    stats = queue.stats()
    assert stats["total"] == 3
    assert stats["by_status"]["done"] == 1
    assert stats["by_status"]["running"] == 1
    assert stats["by_status"]["pending"] == 1

    # Complete remaining
    queue.complete(j1)
    claimed3 = queue.claim_next("worker-2", job_types=["vision"])
    assert claimed3 is not None
    assert claimed3["id"] == j3
    queue.complete(j3)


def test_job_queue_fail_retry_and_stale_recovery(db_conn: sqlite3.Connection) -> None:
    repo = Repository(db_conn)
    queue = JobQueue(db_conn)

    p1 = repo.upsert_photo(picker_id="pick-retry", filename="retry.jpg")
    jid = queue.enqueue("download", p1, max_attempts=2)

    # Attempt 1 -> fail -> back to pending
    job = queue.claim_next("w1")
    assert job is not None and job["attempts"] == 1
    status1 = queue.fail(jid, "network timeout")
    assert status1 == "pending"

    # Attempt 2 -> fail -> permanently failed
    job2 = queue.claim_next("w1")
    assert job2 is not None and job2["attempts"] == 2
    status2 = queue.fail(jid, "expired url")
    assert status2 == "failed"
    assert queue.claim_next("w1") is None

    # Retry failed resets attempts and status
    retried = queue.retry_failed()
    assert retried == 1

    # Claim and simulate stale worker crash
    job3 = queue.claim_next("dead-worker")
    assert job3 is not None
    db_conn.execute(
        "UPDATE jobs SET heartbeat_at = '2020-01-01T00:00:00.000Z' WHERE id = ?",
        (jid,),
    )
    db_conn.commit()

    recovered = queue.claim_next("live-worker", stale_after_seconds=60)
    assert recovered is not None
    assert recovered["id"] == jid
    assert recovered["locked_by"] == "live-worker"
    assert recovered["attempts"] == 2


def test_job_queue_pause_and_resume(db_conn: sqlite3.Connection) -> None:
    repo = Repository(db_conn)
    queue = JobQueue(db_conn)

    p1 = repo.upsert_photo(picker_id="pick-pause", filename="pause.jpg")
    queue.enqueue("vision", p1)

    paused = queue.pause_all("vision")
    assert paused == 1
    assert queue.claim_next("w1", job_types=["vision"]) is None

    resumed = queue.resume_all("vision")
    assert resumed == 1
    claimed = queue.claim_next("w1", job_types=["vision"])
    assert claimed is not None


def test_repository_upsert_dedup_and_search(db_conn: sqlite3.Connection) -> None:
    repo = Repository(db_conn)

    # Dedup on picker_id
    id1 = repo.upsert_photo(
        picker_id="gp-100",
        filename="beach.jpg",
        create_time="2024-07-15T18:30:00Z",
        width=4000,
        height=3000,
    )
    id1_again = repo.upsert_photo(
        picker_id="gp-100",
        filename="beach.jpg",
        camera_make="Google",
        camera_model="Pixel 8",
    )
    assert id1 == id1_again

    id2 = repo.upsert_photo(
        picker_id="gp-200",
        filename="mountain.jpg",
        create_time="2024-08-20T09:15:00Z",
    )
    id3 = repo.upsert_photo(
        picker_id="gp-300",
        filename="winter.jpg",
        create_time="2023-12-25T12:00:00Z",
    )

    repo.mark_downloaded(id1, "data/cache/2024/1.webp", width=1024, height=768)
    repo.mark_downloaded(id2, "data/cache/2024/2.webp")
    repo.mark_downloaded(id3, "data/cache/2023/3.webp")

    repo.save_vision_index(
        photo_id=id1,
        description="Golden sunset over ocean waves at the sandy beach",
        tags=["Beach", "Sunset", "beach", "Ocean"],
        vlm_json={"description": "Golden sunset", "scene": "coastal"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    repo.save_vision_index(
        photo_id=id2,
        description="Snowy alpine mountain peak under clear blue sky",
        tags=["mountain", "hiking", "snow"],
        vlm_json={"description": "Mountain peak", "scene": "alpine"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    repo.save_vision_index(
        photo_id=id3,
        description="Cozy fireplace with hot cocoa in winter cabin",
        tags=["winter", "cabin"],
        vlm_json={"description": "Winter cabin"},
        vlm_model="qwen2.5vl",
        embed_model="nomic-embed-text",
        embedding=[0.0, 0.0, 1.0, 0.0],
    )

    # Verify normalized tags
    photo1 = repo.get_photo(id1)
    assert photo1 is not None
    assert photo1["tags"] == ["beach", "ocean", "sunset"]
    assert photo1["camera_model"] == "Pixel 8"
    assert photo1["status"] == "indexed"

    # FTS5 keyword search
    fts_res = repo.search_photos(text_query="ocean sunset")
    assert [r["id"] for r in fts_res] == [id1]

    # Vector similarity search
    vec_res = repo.search_photos(query_embedding=[0.05, 0.95, 0.0, 0.0], limit=2)
    assert len(vec_res) == 2
    assert vec_res[0]["id"] == id2

    # Hybrid search with year & tag filters
    filtered = repo.search_photos(
        query_embedding=[0.9, 0.1, 0.0, 0.0],
        year=2024,
        tags=["beach"],
    )
    assert [r["id"] for r in filtered] == [id1]

    # Date range filter
    date_res = repo.search_photos(
        start_date="2024-08-01T00:00:00Z",
        end_date="2024-08-31T23:59:59Z",
    )
    assert [r["id"] for r in date_res] == [id2]

    # Year stats
    stats_2024 = repo.get_year_stats(2024)
    assert stats_2024["total_photos"] == 2
    assert stats_2024["indexed_photos"] == 2
    assert len(stats_2024["by_month"]) == 2


def test_groups_crud_reorder_and_repick_manifest(db_conn: sqlite3.Connection) -> None:
    repo = Repository(db_conn)

    p1 = repo.upsert_photo(picker_id="g-1", filename="first.jpg", create_time="2024-05-01T10:00:00Z")
    p2 = repo.upsert_photo(picker_id="g-2", filename="second.jpg", create_time="2024-05-02T10:00:00Z")
    p3 = repo.upsert_photo(picker_id="g-3", filename="third.jpg", create_time="2024-05-03T10:00:00Z")

    repo.mark_downloaded(p1, "data/cache/2024/1.webp")
    repo.mark_original_downloaded(p1, "data/originals/2024/first.jpg")

    gid = repo.create_group("Summer Highlights", year=2024, description="Best shots", kind="manual")
    assert repo.get_group(gid) is not None
    assert len(repo.list_groups(year=2024)) == 1

    repo.add_photos_to_group(gid, [p1, p2, p3], captions={p1: "Opening shot"})
    photos = repo.get_group_photos(gid)
    assert [p["id"] for p in photos] == [p1, p2, p3]
    assert photos[0]["caption"] == "Opening shot"

    # Reorder
    repo.reorder_group(gid, [p3, p1, p2])
    reordered = repo.get_group_photos(gid)
    assert [p["id"] for p in reordered] == [p3, p1, p2]

    # Export repick manifest
    manifest = repo.export_repick_manifest(gid)
    assert [m["photo_id"] for m in manifest] == [p3, p1, p2]
    assert manifest[0]["needs_original"] is True
    assert manifest[1]["needs_original"] is False

    # Remove photo
    removed = repo.remove_photos_from_group(gid, [p2])
    assert removed == 1
    assert [p["id"] for p in repo.get_group_photos(gid)] == [p3, p1]
