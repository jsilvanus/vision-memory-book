"""Worker 2: Vision description and embedding indexer."""

import asyncio
import logging
from pathlib import Path

from vision_memory_book.ai.base import AIProvider
from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
}


class VisionIndexerWorker:
    """Claims 'vision' jobs, runs VLM description + embedding, and indexes into SQLite."""

    def __init__(
        self,
        queue: JobQueue,
        repo: Repository,
        vision_provider: AIProvider,
        embed_provider: AIProvider,
        vlm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.queue = queue
        self.repo = repo
        self.vision_provider = vision_provider
        self.embed_provider = embed_provider
        self.vlm_model = vlm_model or getattr(vision_provider, "vision_model", "unknown")
        self.embed_model = embed_model or getattr(embed_provider, "embed_model", "unknown")

    async def run_once(self, worker_id: str = "vision-worker-1") -> bool:
        """Claim and process a single 'vision' job. Returns True if a job was claimed."""
        job = self.queue.claim_next(worker_id=worker_id, job_types=["vision"])
        if job is None:
            return False

        job_id = int(job["id"])
        photo_id = int(job["photo_id"])

        try:
            photo = self.repo.get_photo(photo_id)
            if photo is None:
                raise ValueError(f"Photo {photo_id} not found")

            cache_path_str = photo.get("cache_path")
            if not cache_path_str:
                raise ValueError(f"Photo {photo_id} has no cache_path")

            cache_path = Path(cache_path_str)
            if not cache_path.is_file():
                raise FileNotFoundError(f"Cached image not found: {cache_path}")

            image_bytes = cache_path.read_bytes()
            mime_type = _MIME_BY_SUFFIX.get(
                cache_path.suffix.lower(),
                photo.get("mime_type") or "image/webp",
            )

            result = await self.vision_provider.describe_image(image_bytes, mime_type)
            self.queue.heartbeat(job_id, worker_id=worker_id)

            embed_input = f"{result.description}\nTags: {', '.join(result.tags)}"
            embeddings = await self.embed_provider.embed([embed_input])
            if not embeddings or not embeddings[0]:
                raise ValueError("Embedding provider returned empty embedding")

            vlm_json = (
                result.raw_json
                if result.raw_json
                else result.model_dump(exclude={"raw_json"})
            )
            self.repo.save_vision_index(
                photo_id=photo_id,
                description=result.description,
                tags=result.tags,
                vlm_json=vlm_json,
                vlm_model=self.vlm_model,
                embed_model=self.embed_model,
                embedding=embeddings[0],
            )
            self.queue.complete(job_id)
            return True
        except Exception as exc:
            logger.exception("Vision indexing failed for job %s (photo %s)", job_id, photo_id)
            status = self.queue.fail(job_id, str(exc))
            if status == "failed":
                self.repo.mark_photo_failed(photo_id)
            return True

    async def run_forever(
        self,
        worker_id: str = "vision-worker-1",
        poll_interval: float = 1.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously claim and process 'vision' jobs until stop_event is set."""
        while stop_event is None or not stop_event.is_set():
            processed = await self.run_once(worker_id=worker_id)
            if not processed:
                if stop_event is not None:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                    except TimeoutError:
                        pass
                else:
                    await asyncio.sleep(poll_interval)
