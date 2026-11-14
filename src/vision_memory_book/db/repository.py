"""Data access layer for photos, tags, FTS5/vector search, groups, and books."""

import json
import re
import sqlite3
from typing import Any

import sqlite_vec


def _extract_year(create_time: str | None) -> int | None:
    if not create_time or len(create_time) < 4:
        return None
    prefix = create_time[:4]
    if prefix.isdigit():
        return int(prefix)
    return None


def _normalize_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        norm = tag.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def _sanitize_fts_query(text_query: str) -> str:
    """Convert user text into a safe FTS5 query string."""
    tokens = re.findall(r"[0-9A-Za-z_]+", text_query)
    if not tokens:
        return ""
    return " OR ".join(f'"{tok}"' for tok in tokens)


class Repository:
    """Repository for all Vision Memory Book persistence and search operations."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # Picker Sessions
    # ------------------------------------------------------------------

    def create_picker_session(
        self,
        session_id: str,
        picker_uri: str,
        expire_time: str | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO picker_sessions (id, picker_uri, status, expire_time)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                picker_uri = excluded.picker_uri,
                status = excluded.status,
                expire_time = excluded.expire_time,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (session_id, picker_uri, status, expire_time),
        )
        self.conn.commit()
        return self.get_picker_session(session_id)  # type: ignore[return-value]

    def update_picker_session_status(self, session_id: str, status: str) -> None:
        self.conn.execute(
            """
            UPDATE picker_sessions
            SET status = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (status, session_id),
        )
        self.conn.commit()

    def get_picker_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM picker_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Photos & Vision Indexing
    # ------------------------------------------------------------------

    def upsert_photo(
        self,
        picker_id: str,
        session_id: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        create_time: str | None = None,
        width: int | None = None,
        height: int | None = None,
        camera_make: str | None = None,
        camera_model: str | None = None,
    ) -> int:
        """Insert or update photo metadata keyed by picker_id, returning photo_id."""
        year = _extract_year(create_time)
        row = self.conn.execute(
            """
            INSERT INTO photos (
                picker_id, session_id, filename, mime_type, create_time,
                year, width, height, camera_make, camera_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(picker_id) DO UPDATE SET
                session_id = COALESCE(excluded.session_id, photos.session_id),
                filename = COALESCE(excluded.filename, photos.filename),
                mime_type = COALESCE(excluded.mime_type, photos.mime_type),
                create_time = COALESCE(excluded.create_time, photos.create_time),
                year = COALESCE(excluded.year, photos.year),
                width = COALESCE(excluded.width, photos.width),
                height = COALESCE(excluded.height, photos.height),
                camera_make = COALESCE(excluded.camera_make, photos.camera_make),
                camera_model = COALESCE(excluded.camera_model, photos.camera_model),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            RETURNING id
            """,
            (
                picker_id,
                session_id,
                filename,
                mime_type,
                create_time,
                year,
                width,
                height,
                camera_make,
                camera_model,
            ),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def mark_downloaded(
        self,
        photo_id: int,
        cache_path: str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE photos
            SET cache_path = ?,
                width = COALESCE(?, width),
                height = COALESCE(?, height),
                status = CASE WHEN status = 'indexed' THEN 'indexed' ELSE 'downloaded' END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (cache_path, width, height, photo_id),
        )
        self.conn.commit()

    def mark_original_downloaded(self, photo_id: int, original_path: str) -> None:
        self.conn.execute(
            """
            UPDATE photos
            SET original_path = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (original_path, photo_id),
        )
        self.conn.commit()

    def mark_photo_failed(self, photo_id: int) -> None:
        self.conn.execute(
            """
            UPDATE photos
            SET status = 'failed',
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (photo_id,),
        )
        self.conn.commit()

    def save_vision_index(
        self,
        photo_id: int,
        description: str,
        tags: list[str],
        vlm_json: dict[str, Any] | str,
        vlm_model: str,
        embed_model: str,
        embedding: list[float],
    ) -> None:
        """Persist VLM description, normalized tags, FTS5 row, and vector embedding."""
        norm_tags = _normalize_tags(tags)
        vlm_json_str = vlm_json if isinstance(vlm_json, str) else json.dumps(vlm_json)
        vec_bytes = sqlite_vec.serialize_float32(embedding)

        photo_row = self.conn.execute(
            "SELECT filename FROM photos WHERE id = ?",
            (photo_id,),
        ).fetchone()
        if photo_row is None:
            raise ValueError(f"Photo {photo_id} not found")
        filename = photo_row["filename"] or ""

        with self.conn:
            self.conn.execute(
                """
                UPDATE photos
                SET description = ?,
                    vlm_json = ?,
                    vlm_model = ?,
                    embed_model = ?,
                    status = 'indexed',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (description, vlm_json_str, vlm_model, embed_model, photo_id),
            )

            self.conn.execute("DELETE FROM photo_tags WHERE photo_id = ?", (photo_id,))
            if norm_tags:
                self.conn.executemany(
                    "INSERT INTO photo_tags (photo_id, tag) VALUES (?, ?)",
                    [(photo_id, tag) for tag in norm_tags],
                )

            self.conn.execute(
                "DELETE FROM photos_fts WHERE photo_id = ?",
                (photo_id,),
            )
            self.conn.execute(
                "INSERT INTO photos_fts (photo_id, filename, description, tags) VALUES (?, ?, ?, ?)",
                (photo_id, filename, description, " ".join(norm_tags)),
            )

            self.conn.execute(
                "DELETE FROM photo_embeddings WHERE photo_id = ?",
                (photo_id,),
            )
            self.conn.execute(
                "INSERT INTO photo_embeddings (photo_id, embedding) VALUES (?, ?)",
                (photo_id, vec_bytes),
            )

    def _get_tags_for_photos(self, photo_ids: list[int]) -> dict[int, list[str]]:
        if not photo_ids:
            return {}
        placeholders = ", ".join("?" for _ in photo_ids)
        rows = self.conn.execute(
            f"SELECT photo_id, tag FROM photo_tags WHERE photo_id IN ({placeholders}) ORDER BY tag ASC",
            photo_ids,
        ).fetchall()
        mapping: dict[int, list[str]] = {pid: [] for pid in photo_ids}
        for row in rows:
            mapping[int(row["photo_id"])].append(row["tag"])
        return mapping

    def get_photo(self, photo_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["tags"] = self._get_tags_for_photos([photo_id]).get(photo_id, [])
        group_rows = self.conn.execute(
            """
            SELECT g.id, g.name, g.year, g.kind, gi.sort_order, gi.caption
            FROM group_items gi
            JOIN groups g ON g.id = gi.group_id
            WHERE gi.photo_id = ?
            ORDER BY g.id ASC
            """,
            (photo_id,),
        ).fetchall()
        data["groups"] = [dict(g) for g in group_rows]
        return data

    def get_photo_by_picker_id(self, picker_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id FROM photos WHERE picker_id = ?",
            (picker_id,),
        ).fetchone()
        if row is None:
            return None
        return self.get_photo(int(row["id"]))

    # ------------------------------------------------------------------
    # Hybrid Search & Stats
    # ------------------------------------------------------------------

    def search_photos(
        self,
        query_embedding: list[float] | None = None,
        text_query: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        year: int | None = None,
        tags: list[str] | None = None,
        group_id: int | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity, FTS5 keyword search, and SQL filters."""
        where_clauses: list[str] = []
        params: list[Any] = []

        if start_date is not None:
            where_clauses.append("p.create_time >= ?")
            params.append(start_date)
        if end_date is not None:
            where_clauses.append("p.create_time <= ?")
            params.append(end_date)
        if year is not None:
            where_clauses.append("p.year = ?")
            params.append(year)
        if status is not None:
            where_clauses.append("p.status = ?")
            params.append(status)
        if group_id is not None:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM group_items gi WHERE gi.photo_id = p.id AND gi.group_id = ?)"
            )
            params.append(group_id)
        if tags:
            norm_tags = _normalize_tags(tags)
            for tag in norm_tags:
                where_clauses.append(
                    "EXISTS (SELECT 1 FROM photo_tags pt WHERE pt.photo_id = p.id AND pt.tag = ?)"
                )
                params.append(tag)

        fts_expr = _sanitize_fts_query(text_query) if text_query else ""

        # Case 1: Vector similarity (optionally combined with FTS5)
        if query_embedding is not None:
            vec_bytes = sqlite_vec.serialize_float32(query_embedding)
            k = max((limit + offset) * 5, 200)
            vec_rows = self.conn.execute(
                """
                SELECT photo_id, distance
                FROM photo_embeddings
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
                """,
                (vec_bytes, k),
            ).fetchall()
            vec_distances: dict[int, float] = {
                int(r["photo_id"]): float(r["distance"]) for r in vec_rows
            }

            fts_ranks: dict[int, float] = {}
            if fts_expr:
                fts_rows = self.conn.execute(
                    """
                    SELECT photo_id, bm25(photos_fts) AS rank
                    FROM photos_fts
                    WHERE photos_fts MATCH ?
                    """,
                    (fts_expr,),
                ).fetchall()
                fts_ranks = {int(r["photo_id"]): float(r["rank"]) for r in fts_rows}

            candidate_ids = list(set(vec_distances.keys()) | set(fts_ranks.keys()))
            if not candidate_ids:
                return []

            placeholders = ", ".join("?" for _ in candidate_ids)
            where_clauses.append(f"p.id IN ({placeholders})")
            params.extend(candidate_ids)

            where_sql = f"WHERE {' AND '.join(where_clauses)}"
            rows = self.conn.execute(
                f"SELECT p.* FROM photos p {where_sql}",
                params,
            ).fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                pid = int(d["id"])
                dist = vec_distances.get(pid, 1e6)
                fts_bonus = 0.15 if pid in fts_ranks else 0.0
                d["distance"] = vec_distances.get(pid)
                d["score"] = dist - fts_bonus
                results.append(d)

            results.sort(key=lambda item: (item["score"], item["id"]))
            sliced = results[offset : offset + limit]
            tag_map = self._get_tags_for_photos([int(r["id"]) for r in sliced])
            for item in sliced:
                item["tags"] = tag_map.get(int(item["id"]), [])
            return sliced

        # Case 2: FTS5 text search only
        if fts_expr:
            where_clauses.append("photos_fts MATCH ?")
            params.append(fts_expr)
            where_sql = f"WHERE {' AND '.join(where_clauses)}"
            sql = f"""
                SELECT p.*, bm25(photos_fts) AS rank
                FROM photos_fts
                JOIN photos p ON p.id = photos_fts.photo_id
                {where_sql}
                ORDER BY rank ASC, p.create_time DESC, p.id DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            rows = self.conn.execute(sql, params).fetchall()
        else:
            # Case 3: Pure SQL filter search
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            sql = f"""
                SELECT p.*
                FROM photos p
                {where_sql}
                ORDER BY p.create_time DESC, p.id DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            rows = self.conn.execute(sql, params).fetchall()

        items = [dict(r) for r in rows]
        tag_map = self._get_tags_for_photos([int(r["id"]) for r in items])
        for item in items:
            item["tags"] = tag_map.get(int(item["id"]), [])
        return items

    def get_year_stats(self, year: int) -> dict[str, Any]:
        """Return photo counts by status, by month, and top tags for a given year."""
        totals = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_photos,
                SUM(CASE WHEN status = 'indexed' THEN 1 ELSE 0 END) AS indexed_photos,
                SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded_photos,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_photos
            FROM photos
            WHERE year = ?
            """,
            (year,),
        ).fetchone()

        month_rows = self.conn.execute(
            """
            SELECT substr(create_time, 1, 7) AS month, COUNT(*) AS count
            FROM photos
            WHERE year = ? AND create_time IS NOT NULL
            GROUP BY month
            ORDER BY month ASC
            """,
            (year,),
        ).fetchall()

        tag_rows = self.conn.execute(
            """
            SELECT pt.tag, COUNT(*) AS count
            FROM photo_tags pt
            JOIN photos p ON p.id = pt.photo_id
            WHERE p.year = ?
            GROUP BY pt.tag
            ORDER BY count DESC, pt.tag ASC
            LIMIT 25
            """,
            (year,),
        ).fetchall()

        return {
            "year": year,
            "total_photos": int(totals["total_photos"] or 0),
            "indexed_photos": int(totals["indexed_photos"] or 0),
            "downloaded_photos": int(totals["downloaded_photos"] or 0),
            "failed_photos": int(totals["failed_photos"] or 0),
            "by_month": [dict(r) for r in month_rows],
            "top_tags": [dict(r) for r in tag_rows],
        }

    # ------------------------------------------------------------------
    # Groups & Group Items
    # ------------------------------------------------------------------

    def create_group(
        self,
        name: str,
        year: int | None = None,
        description: str | None = None,
        kind: str = "manual",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO groups (name, year, description, kind)
            VALUES (?, ?, ?, ?)
            """,
            (name, year, description, kind),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_groups(self, year: int | None = None) -> list[dict[str, Any]]:
        if year is not None:
            rows = self.conn.execute(
                """
                SELECT g.*, COUNT(gi.photo_id) AS photo_count
                FROM groups g
                LEFT JOIN group_items gi ON gi.group_id = g.id
                WHERE g.year = ?
                GROUP BY g.id
                ORDER BY g.created_at DESC, g.id DESC
                """,
                (year,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT g.*, COUNT(gi.photo_id) AS photo_count
                FROM groups g
                LEFT JOIN group_items gi ON gi.group_id = g.id
                GROUP BY g.id
                ORDER BY g.created_at DESC, g.id DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_group(self, group_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT g.*, COUNT(gi.photo_id) AS photo_count
            FROM groups g
            LEFT JOIN group_items gi ON gi.group_id = g.id
            WHERE g.id = ?
            GROUP BY g.id
            """,
            (group_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def add_photos_to_group(
        self,
        group_id: int,
        photo_ids: list[int],
        captions: dict[int, str] | list[str | None] | None = None,
    ) -> int:
        if not photo_ids:
            return 0

        max_row = self.conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) AS max_order FROM group_items WHERE group_id = ?",
            (group_id,),
        ).fetchone()
        next_order = int(max_row["max_order"]) + 1

        added = 0
        with self.conn:
            for idx, photo_id in enumerate(photo_ids):
                caption: str | None = None
                if isinstance(captions, dict):
                    caption = captions.get(photo_id)
                elif isinstance(captions, list) and idx < len(captions):
                    caption = captions[idx]

                cur = self.conn.execute(
                    """
                    INSERT INTO group_items (group_id, photo_id, sort_order, caption)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(group_id, photo_id) DO UPDATE SET
                        caption = COALESCE(excluded.caption, group_items.caption)
                    """,
                    (group_id, photo_id, next_order + idx, caption),
                )
                added += cur.rowcount
        return added

    def remove_photos_from_group(self, group_id: int, photo_ids: list[int]) -> int:
        if not photo_ids:
            return 0
        placeholders = ", ".join("?" for _ in photo_ids)
        cur = self.conn.execute(
            f"DELETE FROM group_items WHERE group_id = ? AND photo_id IN ({placeholders})",
            [group_id, *photo_ids],
        )
        self.conn.commit()
        return cur.rowcount

    def reorder_group(self, group_id: int, ordered_photo_ids: list[int]) -> None:
        with self.conn:
            for order, photo_id in enumerate(ordered_photo_ids):
                self.conn.execute(
                    """
                    UPDATE group_items
                    SET sort_order = ?
                    WHERE group_id = ? AND photo_id = ?
                    """,
                    (order, group_id, photo_id),
                )

    def get_group_photos(self, group_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT p.*, gi.sort_order, gi.caption
            FROM group_items gi
            JOIN photos p ON p.id = gi.photo_id
            WHERE gi.group_id = ?
            ORDER BY gi.sort_order ASC, p.id ASC
            """,
            (group_id,),
        ).fetchall()
        items = [dict(r) for r in rows]
        tag_map = self._get_tags_for_photos([int(r["id"]) for r in items])
        for item in items:
            item["tags"] = tag_map.get(int(item["id"]), [])
        return items

    def export_repick_manifest(self, group_id: int) -> list[dict[str, Any]]:
        """Export checklist of photos in a group for Final-Touch (=d) Google Photos re-pick."""
        rows = self.conn.execute(
            """
            SELECT p.id AS photo_id, p.picker_id, p.filename, p.create_time,
                   p.width, p.height, p.cache_path, p.original_path,
                   gi.sort_order, gi.caption
            FROM group_items gi
            JOIN photos p ON p.id = gi.photo_id
            WHERE gi.group_id = ?
            ORDER BY gi.sort_order ASC, p.id ASC
            """,
            (group_id,),
        ).fetchall()
        manifest: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["needs_original"] = item["original_path"] is None
            manifest.append(item)
        return manifest

    # ------------------------------------------------------------------
    # Books
    # ------------------------------------------------------------------

    def save_book(
        self,
        year: int,
        title: str,
        outline_json: dict[str, Any] | str,
        html_path: str | None = None,
        pdf_path: str | None = None,
    ) -> int:
        outline_str = (
            outline_json if isinstance(outline_json, str) else json.dumps(outline_json)
        )
        cur = self.conn.execute(
            """
            INSERT INTO books (year, title, outline_json, html_path, pdf_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (year, title, outline_str, html_path, pdf_path),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_book(self, book_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_books(self, year: int | None = None) -> list[dict[str, Any]]:
        if year is not None:
            rows = self.conn.execute(
                "SELECT * FROM books WHERE year = ? ORDER BY created_at DESC, id DESC",
                (year,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM books ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]
