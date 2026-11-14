"""Worker 1: SSD-safe resized downloader -> in-memory WebP cache, plus Final-Touch originals."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import re
from typing import Any

import httpx
from PIL import Image, ImageOps

from vision_memory_book.config import Settings, get_settings
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.picker.client import PickerClient

logger = logging.getLogger(__name__)


def _sanitize_filename(filename: str | None, photo_id: int) -> str:
    """Sanitize a remote filename for safe local storage."""
    if not filename:
        return f"photo_{photo_id}.jpg"
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        return f"photo_{photo_id}.jpg"
    return cleaned


def _convert_to_webp_bytes(raw_bytes: bytes, quality: int = 85) -> tuple[bytes, int, int]:
    """Normalize image bytes in memory (EXIF auto-orient) and encode as WebP."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented.mode not in ("RGB", "RGBA"):
            has_alpha = oriented.mode in ("LA", "PA") or (
                oriented.mode == "P" and "transparency" in oriented.info
            )
            oriented = oriented.convert("RGBA" if has_alpha else "RGB")

        width, height = oriented.size
        buf = io.BytesIO()
        oriented.save(buf, format="WEBP", quality=quality, method=4)
        return buf.getvalue(), width, height


class DownloadWorker:
    """Processes `download` and `repick_original` jobs from JobQueue."""

    SUPPORTED_JOB_TYPES = ["download", "repick_original"]

    def __init__(
        self,
        repo: Repository,
        queue: JobQueue,
        picker_client: PickerClient,
        settings: Settings | None = None,
    ) -> None:
        self.repo = repo
        self.queue = queue
        self.picker_client = picker_client
        self.settings = settings or get_settings()

    async def _handle_download(self, job: dict[str, Any]) -> None:
        photo_id = int(job["photo_id"])
        payload = job.get("payload") or {}
        base_url = payload.get("base_url")
        if not base_url:
            raise ValueError(f"Download job {job['id']} missing base_url in payload")

        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Photo {photo_id} not found")

        raw_bytes = await self.picker_client.download_media_bytes(
            base_url=base_url,
            max_dim=self.settings.cache_max_dim,
            original=False,
        )
        webp_bytes, width, height = _convert_to_webp_bytes(raw_bytes)

        year_dir = str(photo.get("year") or "unknown")
        out_path = self.settings.cache_dir / year_dir / f"{photo_id}.webp"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(webp_bytes)

        self.repo.mark_downloaded(
            photo_id=photo_id,
            cache_path=str(out_path),
            width=width,
            height=height,
        )
        self.queue.enqueue("vision", photo_id)

    async def _handle_repick_original(self, job: dict[str, Any]) -> None:
        photo_id = int(job["photo_id"])
        payload = job.get("payload") or {}
        base_url = payload.get("base_url")
        if not base_url:
            raise ValueError(f"repick_original job {job['id']} missing base_url in payload")

        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Photo {photo_id} not found")

        raw_bytes = await self.picker_client.download_media_bytes(
            base_url=base_url,
            original=True,
        )

        safe_filename = _sanitize_filename(
            payload.get("filename") or photo.get("filename"),
            photo_id=photo_id,
        )
        out_path = self.settings.originals_dir / f"{photo_id}_{safe_filename}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw_bytes)

        self.repo.mark_original_downloaded(photo_id=photo_id, original_path=str(out_path))

    async def run_once(
        self,
        worker_id: str = "downloader-1",
        stale_after_seconds: int = 300,
    ) -> bool:
        """Claim and process a single download or repick_original job. Returns True if a job was claimed."""
        job = self.queue.claim_next(
            worker_id=worker_id,
            job_types=self.SUPPORTED_JOB_TYPES,
            stale_after_seconds=stale_after_seconds,
        )
        if job is None:
            return False

        job_id = int(job["id"])
        job_type = job["job_type"]
        photo_id = int(job["photo_id"])

        try:
            if job_type == "download":
                await self._handle_download(job)
            elif job_type == "repick_original":
                await self._handle_repick_original(job)
            else:
                raise ValueError(f"Unsupported job_type: {job_type}")
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in (401, 403):
                err_msg = f"HTTP {status_code} expired or unauthorized baseUrl: {exc}"
            else:
                err_msg = f"HTTP {status_code} downloading media: {exc}"
            logger.warning("Job %s (%s) failed: %s", job_id, job_type, err_msg)
            new_status = self.queue.fail(job_id, err_msg)
            if new_status == "failed" and job_type == "download":
                self.repo.mark_photo_failed(photo_id)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Job %s (%s) failed: %s", job_id, job_type, err_msg)
            new_status = self.queue.fail(job_id, err_msg)
            if new_status == "failed" and job_type == "download":
                self.repo.mark_photo_failed(photo_id)
        else:
            self.queue.complete(job_id)

        return True

    async def run_forever(
        self,
        worker_id: str = "downloader-1",
        poll_interval: float = 1.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously poll and process download jobs until stop_event is set."""
        while stop_event is None or not stop_event.is_set():
            processed = await self.run_once(worker_id=worker_id)
            if not processed:
                if stop_event is None:
                    await asyncio.sleep(poll_interval)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                    except TimeoutError:
                        pass
