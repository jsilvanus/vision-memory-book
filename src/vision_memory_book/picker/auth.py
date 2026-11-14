"""Google OAuth2 Desktop/Loopback flow and token persistence for Photos Picker API."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import httpx

from vision_memory_book.config import Settings, get_settings

PICKER_SCOPE = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class OAuthError(RuntimeError):
    """Raised when Google OAuth2 authorization or token refresh fails."""


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE (code_verifier, code_challenge) S256 pair."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class GoogleOAuthManager:
    """Manages Google OAuth2 Desktop/Loopback tokens with persistence and auto-refresh."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_path: Path | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client
        self.client_id = client_id if client_id is not None else self.settings.google_client_id
        self.client_secret = (
            client_secret if client_secret is not None else self.settings.google_client_secret
        )
        self.token_path = Path(token_path or self.settings.google_token_path)
        self.scopes = scopes or [PICKER_SCOPE]
        self._token_cache: dict[str, Any] | None = None

    def load_token(self) -> dict[str, Any] | None:
        """Load persisted OAuth token JSON from disk if present."""
        if self._token_cache is not None:
            return self._token_cache
        if not self.token_path.exists():
            return None
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OAuthError(f"Failed to read token file {self.token_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise OAuthError(f"Invalid token format in {self.token_path}")
        self._token_cache = data
        return data

    def save_token(self, token_data: dict[str, Any]) -> None:
        """Persist OAuth token JSON to disk with restrictive permissions."""
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
        try:
            os.chmod(self.token_path, 0o600)
        except OSError:
            pass
        self._token_cache = token_data

    @staticmethod
    def is_expired(token_data: dict[str, Any], leeway_seconds: int = 60) -> bool:
        """Return True if the access token is missing or past its expiry window."""
        if not token_data.get("access_token"):
            return True
        expires_at = token_data.get("expires_at")
        if expires_at is None:
            return False
        return time.time() + leeway_seconds >= float(expires_at)

    def build_authorization_url(
        self,
        redirect_uri: str,
        state: str,
        code_challenge: str | None = None,
    ) -> str:
        """Construct the Google OAuth2 authorization URL."""
        if not self.client_id:
            raise OAuthError("google_client_id is not configured")
        params: dict[str, str] = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def _post_token(self, payload: dict[str, str]) -> dict[str, Any]:
        """POST to Google's OAuth2 token endpoint and normalize the response."""
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(GOOGLE_TOKEN_URL, data=payload)
            if resp.status_code >= 400:
                raise OAuthError(
                    f"Google token endpoint returned {resp.status_code}: {resp.text}"
                )
            data = resp.json()
        except httpx.HTTPError as exc:
            raise OAuthError(f"Token request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

        if "access_token" not in data:
            raise OAuthError(f"Token response missing access_token: {data}")

        expires_in = float(data.get("expires_in", 3600))
        data["expires_at"] = time.time() + expires_in
        return data

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any]:
        """Exchange an authorization code for access and refresh tokens."""
        if not self.client_id:
            raise OAuthError("google_client_id is not configured")
        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
        }
        if self.client_secret:
            payload["client_secret"] = self.client_secret
        if code_verifier:
            payload["code_verifier"] = code_verifier

        token_data = await self._post_token(payload)
        self.save_token(token_data)
        return token_data

    async def refresh_access_token(self, refresh_token: str | None = None) -> dict[str, Any]:
        """Refresh an expired access token using the stored or provided refresh_token."""
        existing = self.load_token() or {}
        rtok = refresh_token or existing.get("refresh_token")
        if not rtok:
            raise OAuthError("No refresh_token available; re-authorization required")
        if not self.client_id:
            raise OAuthError("google_client_id is not configured")

        payload: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": str(rtok),
            "client_id": self.client_id,
        }
        if self.client_secret:
            payload["client_secret"] = self.client_secret

        refreshed = await self._post_token(payload)
        if "refresh_token" not in refreshed:
            refreshed["refresh_token"] = rtok
        self.save_token(refreshed)
        return refreshed

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing automatically if expired."""
        token_data = self.load_token()
        if token_data is None:
            raise OAuthError(
                f"No OAuth token found at {self.token_path}; run OAuth authorization first."
            )
        if self.is_expired(token_data):
            token_data = await self.refresh_access_token(token_data.get("refresh_token"))
        return str(token_data["access_token"])

    async def authorize(
        self,
        open_browser: bool = True,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Run the OAuth2 Desktop loopback flow and persist the resulting token."""
        state = secrets.token_urlsafe(32)
        verifier, challenge = _generate_pkce_pair()
        loop = asyncio.get_running_loop()
        auth_result: asyncio.Future[tuple[str, str]] = loop.create_future()

        async def handle_callback(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                raw_request = await reader.read(4096)
                request_line = raw_request.decode("utf-8", errors="replace").splitlines()[0]
                target = request_line.split(" ")[1] if " " in request_line else "/"
                parsed = urlparse(target)
                query = parse_qs(parsed.query)

                if "error" in query:
                    err = query["error"][0]
                    if not auth_result.done():
                        auth_result.set_exception(OAuthError(f"OAuth error: {err}"))
                    body = "Authorization failed. You may close this window."
                    status = "400 Bad Request"
                elif "code" in query and "state" in query:
                    code = query["code"][0]
                    recv_state = query["state"][0]
                    if not auth_result.done():
                        auth_result.set_result((code, recv_state))
                    body = "Authorization successful! You may close this window."
                    status = "200 OK"
                else:
                    body = "Missing authorization code."
                    status = "400 Bad Request"

                response = (
                    f"HTTP/1.1 {status}\r\n"
                    "Content-Type: text/plain; charset=utf-8\r\n"
                    f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                    "Connection: close\r\n\r\n"
                    f"{body}"
                )
                writer.write(response.encode("utf-8"))
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle_callback, "127.0.0.1", 0)
        sock = server.sockets[0]
        port = sock.getsockname()[1]
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        auth_url = self.build_authorization_url(
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=challenge,
        )

        if open_browser:
            webbrowser.open(auth_url)

        try:
            async with server:
                code, recv_state = await asyncio.wait_for(auth_result, timeout=timeout)
        except TimeoutError as exc:
            raise OAuthError("Timed out waiting for OAuth loopback callback") from exc

        if not secrets.compare_digest(recv_state, state):
            raise OAuthError("OAuth state mismatch")

        return await self.exchange_code(
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
        )
