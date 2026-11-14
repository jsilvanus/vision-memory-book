-- Vision Memory Book SQLite Schema
-- Requires SQLite with FTS5 and sqlite-vec extension loaded.

CREATE TABLE IF NOT EXISTS picker_sessions (
    id TEXT PRIMARY KEY,
    picker_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'expired')),
    expire_time TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    picker_id TEXT NOT NULL UNIQUE,
    session_id TEXT REFERENCES picker_sessions(id) ON DELETE SET NULL,
    filename TEXT,
    mime_type TEXT,
    create_time TEXT,
    year INTEGER,
    width INTEGER,
    height INTEGER,
    camera_make TEXT,
    camera_model TEXT,
    cache_path TEXT,
    original_path TEXT,
    description TEXT,
    vlm_json TEXT,
    vlm_model TEXT,
    embed_model TEXT,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'downloaded', 'indexed', 'failed')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_photos_year ON photos(year);
CREATE INDEX IF NOT EXISTS idx_photos_create_time ON photos(create_time);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);

CREATE TABLE IF NOT EXISTS photo_tags (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (photo_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag);

CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    photo_id UNINDEXED,
    filename,
    description,
    tags
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed', 'paused')),
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    locked_by TEXT,
    locked_at TEXT,
    heartbeat_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, job_type, priority DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_photo ON jobs(photo_id, job_type);

CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    year INTEGER,
    description TEXT,
    kind TEXT NOT NULL DEFAULT 'manual' CHECK (kind IN ('manual', 'book_chapter')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS group_items (
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    caption TEXT,
    PRIMARY KEY (group_id, photo_id)
);

CREATE INDEX IF NOT EXISTS idx_group_items_order ON group_items(group_id, sort_order, photo_id);

CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    title TEXT NOT NULL,
    outline_json TEXT NOT NULL DEFAULT '{}',
    html_path TEXT,
    pdf_path TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
