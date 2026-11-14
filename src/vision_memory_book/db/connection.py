"""SQLite connection management with WAL mode and sqlite-vec."""

from pathlib import Path
import sqlite3

import sqlite_vec


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL, foreign keys, and sqlite-vec."""
    db_str = str(db_path)
    if db_str != ":memory:" and not db_str.startswith("file:"):
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(db_path: str | Path, embed_dim: int = 768) -> sqlite3.Connection:
    """Initialize database schema and sqlite-vec photo_embeddings virtual table."""
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")

    conn = get_connection(db_path)
    schema_path = Path(__file__).with_name("schema.sql")
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS photo_embeddings "
        f"USING vec0(photo_id INTEGER PRIMARY KEY, embedding float[{int(embed_dim)}]);"
    )
    conn.commit()
    return conn
