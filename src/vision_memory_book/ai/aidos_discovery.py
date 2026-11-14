"""Aidos RFC-0103 loopback endpoint discovery."""

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class AidosEndpoint:
    """Resolved Aidos OpenAI-compatible endpoint and optional bearer token."""

    base_url: str
    token: str | None = None


def _ensure_v1_suffix(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def discover_aidos_endpoint(
    discovery_path: str | Path | None = None,
    default_port: int = 18080,
) -> AidosEndpoint:
    """Resolve Aidos RFC-0103 loopback endpoint from a discovery JSON file or default port.

    Supports discovery JSON containing either ``base_url`` or ``port`` (with optional
    ``host``), and ``token`` or ``bearer_token``. Ensures ``/v1`` suffix on ``base_url``.
    Falls back cleanly to ``http://127.0.0.1:{default_port}/v1`` if no file exists.
    """
    fallback = AidosEndpoint(base_url=f"http://127.0.0.1:{default_port}/v1", token=None)

    candidate: Path | None = None
    if discovery_path is not None:
        candidate = Path(discovery_path).expanduser()
    else:
        env_path = os.environ.get("AIDOS_DISCOVERY_PATH")
        if env_path:
            candidate = Path(env_path).expanduser()

    if candidate is None or not candidate.is_file():
        return fallback

    try:
        raw = candidate.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return fallback

    if not isinstance(data, dict):
        return fallback

    raw_token = data.get("token") or data.get("bearer_token")
    token = str(raw_token).strip() if raw_token else None
    if token == "":
        token = None

    raw_base_url = data.get("base_url")
    if isinstance(raw_base_url, str) and raw_base_url.strip():
        return AidosEndpoint(base_url=_ensure_v1_suffix(raw_base_url), token=token)

    raw_port = data.get("port")
    if raw_port is not None:
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = default_port
        host = str(data.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        return AidosEndpoint(base_url=f"http://{host}:{port}/v1", token=token)

    return AidosEndpoint(base_url=fallback.base_url, token=token)
