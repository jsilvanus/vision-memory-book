"""FastAPI web UI and export endpoints for Vision Memory Book."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import zipfile

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from vision_memory_book.config import Settings, get_settings
from vision_memory_book.db.connection import init_db
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    accept = request.headers.get("accept", "")
    return "application/json" in content_type or (
        "application/json" in accept and "text/html" not in accept
    )


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid integer value: {value!r}") from exc


def _parse_int_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        result: list[int] = []
        for item in raw:
            parsed = _parse_optional_int(item)
            if parsed is not None:
                result.append(parsed)
        return result
    s = str(raw).strip()
    if not s:
        return []
    result = []
    for part in s.split(","):
        parsed = _parse_optional_int(part)
        if parsed is not None:
            result.append(parsed)
    return result


def create_app(
    settings: Settings | None = None,
    repo: Repository | None = None,
    queue: JobQueue | None = None,
) -> FastAPI:
    """Create and configure the Vision Memory Book FastAPI application."""
    cfg = settings or get_settings()

    if repo is None and queue is None:
        cfg.ensure_dirs()
        conn = init_db(cfg.db_path, embed_dim=cfg.embed_dim)
        repo = Repository(conn)
        queue = JobQueue(conn)
    elif repo is not None and queue is None:
        queue = JobQueue(repo.conn)
    elif queue is not None and repo is None:
        repo = Repository(queue.conn)

    assert repo is not None and queue is not None

    app = FastAPI(title="Vision Memory Book", version="0.1.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.state.settings = cfg
    app.state.repo = repo
    app.state.queue = queue

    # ------------------------------------------------------------------
    # Dashboard & Queue Control
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, year: str | None = Query(None)) -> HTMLResponse:
        queue_stats = queue.stats()
        year_rows = repo.conn.execute(
            "SELECT DISTINCT year FROM photos WHERE year IS NOT NULL ORDER BY year DESC"
        ).fetchall()
        available_years = [int(r["year"]) for r in year_rows]

        selected_year = _parse_optional_int(year)
        if selected_year is not None:
            year_stats_list = [repo.get_year_stats(selected_year)]
        else:
            year_stats_list = [repo.get_year_stats(y) for y in available_years]

        recent_groups = repo.list_groups()[:10]
        total_photos_row = repo.conn.execute("SELECT COUNT(*) AS cnt FROM photos").fetchone()
        total_photos = int(total_photos_row["cnt"]) if total_photos_row else 0

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "queue_stats": queue_stats,
                "available_years": available_years,
                "selected_year": selected_year,
                "year_stats_list": year_stats_list,
                "recent_groups": recent_groups,
                "total_photos": total_photos,
            },
        )

    @app.post("/queue/control")
    async def queue_control(request: Request):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            action = str(body.get("action", "")).strip()
            job_type = body.get("job_type") or None
        else:
            form = await request.form()
            action = str(form.get("action", "")).strip()
            raw_job_type = form.get("job_type")
            job_type = str(raw_job_type).strip() if raw_job_type else None

        if action == "pause":
            affected = queue.pause_all(job_type=job_type)
        elif action == "resume":
            affected = queue.resume_all(job_type=job_type)
        elif action == "retry_failed":
            affected = queue.retry_failed(job_type=job_type)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid queue action {action!r}. Must be 'pause', 'resume', or 'retry_failed'.",
            )

        if _is_json_request(request):
            return JSONResponse(
                {"action": action, "affected": affected, "stats": queue.stats()}
            )
        return RedirectResponse(url="/", status_code=303)

    # ------------------------------------------------------------------
    # Gallery & Photo Serving
    # ------------------------------------------------------------------

    @app.get("/gallery", response_class=HTMLResponse)
    def gallery(
        request: Request,
        q: str | None = Query(None),
        year: str | None = Query(None),
        start_date: str | None = Query(None),
        end_date: str | None = Query(None),
        tag: str | None = Query(None),
        limit: int = Query(60, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> HTMLResponse:
        parsed_year = _parse_optional_int(year)
        clean_q = q.strip() if q and q.strip() else None
        clean_start = start_date.strip() if start_date and start_date.strip() else None
        clean_end = end_date.strip() if end_date and end_date.strip() else None
        clean_tag = tag.strip().lower() if tag and tag.strip() else None

        photos = repo.search_photos(
            text_query=clean_q,
            year=parsed_year,
            start_date=clean_start,
            end_date=clean_end,
            tags=[clean_tag] if clean_tag else None,
            limit=limit,
            offset=offset,
        )
        groups = repo.list_groups()
        year_rows = repo.conn.execute(
            "SELECT DISTINCT year FROM photos WHERE year IS NOT NULL ORDER BY year DESC"
        ).fetchall()
        available_years = [int(r["year"]) for r in year_rows]

        return templates.TemplateResponse(
            request,
            "gallery.html",
            {
                "photos": photos,
                "groups": groups,
                "available_years": available_years,
                "filters": {
                    "q": clean_q or "",
                    "year": parsed_year,
                    "start_date": clean_start or "",
                    "end_date": clean_end or "",
                    "tag": clean_tag or "",
                },
                "limit": limit,
                "offset": offset,
            },
        )

    @app.get("/photos/{photo_id}/image")
    def serve_photo_image(
        photo_id: int,
        prefer_original: bool = Query(False),
    ) -> FileResponse:
        photo = repo.get_photo(photo_id)
        if photo is None:
            raise HTTPException(status_code=404, detail="Photo not found")

        if prefer_original and photo.get("original_path"):
            orig = Path(photo["original_path"])
            if orig.is_file():
                return FileResponse(path=orig)

        if photo.get("cache_path"):
            cached = Path(photo["cache_path"])
            if cached.is_file():
                return FileResponse(path=cached)

        raise HTTPException(status_code=404, detail="Photo image file not found")

    # ------------------------------------------------------------------
    # Groups & Group Items
    # ------------------------------------------------------------------

    @app.get("/groups", response_class=HTMLResponse)
    def list_groups_page(request: Request, year: str | None = Query(None)) -> HTMLResponse:
        parsed_year = _parse_optional_int(year)
        groups = repo.list_groups(year=parsed_year)
        return templates.TemplateResponse(
            request,
            "groups.html",
            {
                "groups": groups,
                "selected_year": parsed_year,
            },
        )

    @app.post("/groups")
    async def create_group(request: Request):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            name = str(body.get("name", "")).strip()
            year = _parse_optional_int(body.get("year"))
            description = body.get("description")
            kind = str(body.get("kind") or "manual").strip()
        else:
            form = await request.form()
            name = str(form.get("name", "")).strip()
            year = _parse_optional_int(form.get("year"))
            raw_desc = form.get("description")
            description = str(raw_desc).strip() if raw_desc and str(raw_desc).strip() else None
            raw_kind = form.get("kind")
            kind = str(raw_kind).strip() if raw_kind and str(raw_kind).strip() else "manual"

        if not name:
            raise HTTPException(status_code=400, detail="Group name is required")
        if kind not in {"manual", "book_chapter"}:
            raise HTTPException(status_code=400, detail=f"Invalid group kind: {kind!r}")

        group_id = repo.create_group(name=name, year=year, description=description, kind=kind)
        if _is_json_request(request):
            return JSONResponse(
                {"id": group_id, "group": repo.get_group(group_id)},
                status_code=201,
            )
        return RedirectResponse(url=f"/groups/{group_id}", status_code=303)

    @app.get("/groups/{group_id}", response_class=HTMLResponse)
    def group_detail(request: Request, group_id: int) -> HTMLResponse:
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")
        photos = repo.get_group_photos(group_id)
        return templates.TemplateResponse(
            request,
            "group_detail.html",
            {
                "group": group,
                "photos": photos,
            },
        )

    @app.post("/groups/{group_id}/items")
    async def mutate_group_items(request: Request, group_id: int):
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            action = str(body.get("action") or "add").strip()
            if action == "add":
                photo_ids = _parse_int_list(body.get("photo_ids"))
                captions = body.get("captions")
                if isinstance(captions, dict):
                    captions = {int(k): v for k, v in captions.items()}
                repo.add_photos_to_group(group_id, photo_ids, captions=captions)
            elif action == "remove":
                photo_ids = _parse_int_list(body.get("photo_ids"))
                if not photo_ids and body.get("photo_id") is not None:
                    pid = _parse_optional_int(body.get("photo_id"))
                    if pid is not None:
                        photo_ids = [pid]
                repo.remove_photos_from_group(group_id, photo_ids)
            elif action == "reorder":
                ordered_ids = _parse_int_list(body.get("ordered_photo_ids"))
                repo.reorder_group(group_id, ordered_ids)
            elif action == "update_caption":
                pid = _parse_optional_int(body.get("photo_id"))
                if pid is None:
                    raise HTTPException(status_code=400, detail="photo_id is required")
                caption = str(body.get("caption") or "")
                repo.add_photos_to_group(group_id, [pid], captions={pid: caption})
            else:
                raise HTTPException(status_code=400, detail=f"Invalid action: {action!r}")
        else:
            form = await request.form()
            action = str(form.get("action") or "add").strip()
            if action == "add":
                raw_ids = form.getlist("photo_ids")
                photo_ids = _parse_int_list(raw_ids)
                raw_caption = form.get("caption")
                captions = (
                    {pid: str(raw_caption) for pid in photo_ids}
                    if raw_caption is not None and str(raw_caption).strip()
                    else None
                )
                repo.add_photos_to_group(group_id, photo_ids, captions=captions)
            elif action == "remove":
                raw_ids = form.getlist("photo_ids")
                photo_ids = _parse_int_list(raw_ids)
                if not photo_ids and form.get("photo_id") is not None:
                    pid = _parse_optional_int(form.get("photo_id"))
                    if pid is not None:
                        photo_ids = [pid]
                repo.remove_photos_from_group(group_id, photo_ids)
            elif action == "reorder":
                raw_ordered = form.getlist("ordered_photo_ids")
                ordered_ids = (
                    _parse_int_list(raw_ordered[0])
                    if len(raw_ordered) == 1
                    else _parse_int_list(raw_ordered)
                )
                repo.reorder_group(group_id, ordered_ids)
            elif action == "update_caption":
                pid = _parse_optional_int(form.get("photo_id"))
                if pid is None:
                    raise HTTPException(status_code=400, detail="photo_id is required")
                caption = str(form.get("caption") or "")
                repo.add_photos_to_group(group_id, [pid], captions={pid: caption})
            else:
                raise HTTPException(status_code=400, detail=f"Invalid action: {action!r}")

        if _is_json_request(request):
            return JSONResponse(
                {
                    "ok": True,
                    "group_id": group_id,
                    "photos": repo.get_group_photos(group_id),
                }
            )
        return RedirectResponse(url=f"/groups/{group_id}", status_code=303)

    # ------------------------------------------------------------------
    # ZIP Export
    # ------------------------------------------------------------------

    @app.get("/groups/{group_id}/download.zip")
    def download_group_zip(group_id: int) -> StreamingResponse:
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        photos = repo.get_group_photos(group_id)
        buf = io.BytesIO()
        metadata_items: list[dict[str, Any]] = []

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, photo in enumerate(photos, start=1):
                orig_path = Path(photo["original_path"]) if photo.get("original_path") else None
                cache_path = Path(photo["cache_path"]) if photo.get("cache_path") else None

                chosen_path: Path | None = None
                used_original = False
                if orig_path is not None and orig_path.is_file():
                    chosen_path = orig_path
                    used_original = True
                elif cache_path is not None and cache_path.is_file():
                    chosen_path = cache_path

                archive_filename: str | None = None
                if chosen_path is not None:
                    orig_stem = Path(photo.get("filename") or f"photo_{photo['id']}").stem
                    safe_stem = "".join(
                        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in orig_stem
                    )
                    archive_filename = f"{idx:03d}_{safe_stem}{chosen_path.suffix}"
                    zf.write(chosen_path, arcname=archive_filename)

                metadata_items.append(
                    {
                        "photo_id": photo["id"],
                        "picker_id": photo["picker_id"],
                        "filename": photo.get("filename"),
                        "archive_filename": archive_filename,
                        "create_time": photo.get("create_time"),
                        "caption": photo.get("caption"),
                        "description": photo.get("description"),
                        "sort_order": photo.get("sort_order"),
                        "tags": photo.get("tags", []),
                        "used_original": used_original,
                    }
                )

            metadata_payload = {
                "group": {
                    "id": group["id"],
                    "name": group["name"],
                    "year": group.get("year"),
                    "description": group.get("description"),
                    "kind": group.get("kind"),
                },
                "items": metadata_items,
            }
            zf.writestr("metadata.json", json.dumps(metadata_payload, indent=2))

        buf.seek(0)
        headers = {
            "Content-Disposition": f'attachment; filename="group-{group_id}.zip"',
        }
        return StreamingResponse(buf, media_type="application/zip", headers=headers)

    # ------------------------------------------------------------------
    # Final-Touch Re-Pick Workflow
    # ------------------------------------------------------------------

    @app.get("/groups/{group_id}/repick", response_class=HTMLResponse)
    def repick_page(
        request: Request,
        group_id: int,
        matched: int | None = Query(None),
    ) -> HTMLResponse:
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        manifest = repo.export_repick_manifest(group_id)
        missing_originals = [item for item in manifest if item["needs_original"]]
        completed_originals = [item for item in manifest if not item["needs_original"]]

        return templates.TemplateResponse(
            request,
            "repick.html",
            {
                "group": group,
                "missing_originals": missing_originals,
                "completed_originals": completed_originals,
                "total_count": len(manifest),
                "matched_count": matched,
            },
        )

    @app.get("/groups/{group_id}/repick/manifest.json")
    def repick_manifest_json(group_id: int) -> JSONResponse:
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        manifest = repo.export_repick_manifest(group_id)
        missing_count = sum(1 for item in manifest if item["needs_original"])
        return JSONResponse(
            content={
                "group_id": group_id,
                "group_name": group["name"],
                "total_count": len(manifest),
                "missing_count": missing_count,
                "items": manifest,
            }
        )

    @app.post("/groups/{group_id}/repick/match")
    async def repick_match(request: Request, group_id: int):
        group = repo.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            raw_body = await request.json()
            if isinstance(raw_body, list):
                picked_items = raw_body
            elif isinstance(raw_body, dict):
                picked_items = raw_body.get("items", [])
            else:
                raise HTTPException(status_code=400, detail="Expected JSON list or {items: [...]}")
        else:
            form = await request.form()
            items_json = str(form.get("items_json") or "[]").strip()
            try:
                parsed = json.loads(items_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
            if isinstance(parsed, list):
                picked_items = parsed
            elif isinstance(parsed, dict):
                picked_items = parsed.get("items", [])
            else:
                raise HTTPException(status_code=400, detail="Expected JSON list of picked items")

        manifest = repo.export_repick_manifest(group_id)
        by_picker_id = {item["picker_id"]: item for item in manifest}

        matched: list[dict[str, Any]] = []
        unmatched: list[str] = []

        for entry in picked_items:
            if not isinstance(entry, dict):
                continue
            picker_id = str(entry.get("picker_id") or "").strip()
            base_url = str(entry.get("base_url") or "").strip()
            if not picker_id:
                continue

            target = by_picker_id.get(picker_id)
            if target is not None and base_url:
                job_id = queue.enqueue(
                    "repick_original",
                    int(target["photo_id"]),
                    payload={
                        "base_url": base_url,
                        "picker_id": picker_id,
                        "group_id": group_id,
                    },
                    priority=10,
                )
                matched.append(
                    {
                        "photo_id": int(target["photo_id"]),
                        "picker_id": picker_id,
                        "job_id": job_id,
                    }
                )
            else:
                unmatched.append(picker_id)

        if _is_json_request(request):
            return JSONResponse(
                {
                    "group_id": group_id,
                    "matched_count": len(matched),
                    "matched": matched,
                    "unmatched": unmatched,
                }
            )
        return RedirectResponse(
            url=f"/groups/{group_id}/repick?matched={len(matched)}",
            status_code=303,
        )

    return app
