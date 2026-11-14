"""Async Google Photos Picker API client and session ingestion helper."""

from __future__ import annotations

from typing import Any

import httpx

from vision_memory_book.db.queue import JobQueue
from vision_memory_book.db.repository import Repository
from vision_memory_book.picker.auth import GoogleOAuthManager

PICKER_BASE_URL = "https://photospicker.googleapis.com/v1"


class PickerClient:
    """Async client for the Google Photos Picker API."""

    def __init__(
        self,
        auth_manager: GoogleOAuthManager | None = None,
        http_client: httpx.AsyncClient | None = None,
        access_token: str | None = None,
        base_url: str = PICKER_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.auth_manager = auth_manager
        self._access_token = access_token
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> PickerClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await aclose_if_owned(self)

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this PickerClient."""
        if self._owns_client:
            await self.http_client.aclose()

    async def _auth_headers(self) -> dict[str, str]:
        if self._access_token is not None:
            token = self._access_token
        elif self.auth_manager is not None:
            token = await self.auth_manager.get_access_token()
        else:
            raise RuntimeError("PickerClient requires either auth_manager or access_token")
        return {"Authorization": f"Bearer {token}"}

    async def create_session(self) -> dict[str, Any]:
        """Create a new Google Photos Picker session."""
        headers = await self._auth_headers()
        resp = await self.http_client.post(
            f"{self.base_url}/sessions",
            headers=headers,
            json={},
        )
        resp.raise_for_status()
        return dict(resp.json())

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Fetch current status/metadata for a Google Photos Picker session."""
        headers = await self._auth_headers()
        resp = await self.http_client.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=headers,
        )
        resp.raise_for_status()
        return dict(resp.json())

    async def delete_session(self, session_id: str) -> None:
        """Delete a Google Photos Picker session."""
        headers = await self._auth_headers()
        resp = await self.http_client.delete(
            f"{self.base_url}/sessions/{session_id}",
            headers=headers,
        )
        resp.raise_for_status()

    async def list_media_items(
        self,
        session_id: str,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """List all PickedMediaItems in a session, following nextPageToken pagination."""
        items: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            headers = await self._auth_headers()
            params: dict[str, Any] = {
                "sessionId": session_id,
                "pageSize": page_size,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await self.http_client.get(
                f"{self.base_url}/mediaItems",
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("mediaItems") or []
            items.extend(batch)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return items

    async def download_media_bytes(
        self,
        base_url: str,
        max_dim: int | None = None,
        original: bool = False,
    ) -> bytes:
        """Download resized (=wN-hN) or original (=d) bytes from a Picker baseUrl."""
        if original:
            download_url = f"{base_url}=d"
        else:
            dim = max_dim or 1024
            download_url = f"{base_url}=w{dim}-h{dim}"

        headers = await self._auth_headers()
        resp = await self.http_client.get(download_url, headers=headers)
        resp.raise_for_status()
        return resp.content


async def aclose_if_owned(client: PickerClient) -> None:
    await client.aclose()


async def ingest_picker_session(
    client: PickerClient,
    repo: Repository,
    queue: JobQueue,
    session_id: str,
    max_attempts: int = 3,
) -> dict[str, int]:
    """List items in a Picker session, upsert photos, and enqueue missing downloads."""
    if repo.get_picker_session(session_id) is None:
        repo.create_picker_session(
            session_id=session_id,
            picker_uri=f"https://photos.google.com/picker/{session_id}",
            status="completed",
        )

    items = await client.list_media_items(session_id)
    enqueued = 0
    skipped = 0

    for item in items:
        picker_id = item.get("id")
        if not picker_id:
            skipped += 1
            continue

        media_file = item.get("mediaFile") or {}
        meta = media_file.get("mediaFileMetadata") or {}
        base_url = media_file.get("baseUrl") or item.get("baseUrl")
        filename = media_file.get("filename") or item.get("filename")
        mime_type = media_file.get("mimeType") or item.get("mimeType")
        create_time = item.get("createTime")

        photo_id = repo.upsert_photo(
            picker_id=picker_id,
            session_id=session_id,
            filename=filename,
            mime_type=mime_type,
            create_time=create_time,
            width=meta.get("width"),
            height=meta.get("height"),
            camera_make=meta.get("cameraMake"),
            camera_model=meta.get("cameraModel"),
        )

        photo = repo.get_photo(photo_id)
        if photo is not None and photo["status"] in ("downloaded", "indexed"):
            skipped += 1
            continue

        if not base_url:
            skipped += 1
            continue

        queue.enqueue(
            job_type="download",
            photo_id=photo_id,
            payload={
                "base_url": base_url,
                "picker_id": picker_id,
                "session_id": session_id,
                "filename": filename,
            },
            max_attempts=max_attempts,
        )
        enqueued += 1

    return {
        "total": len(items),
        "enqueued": enqueued,
        "skipped": skipped,
    }
