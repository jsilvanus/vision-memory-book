"""OpenAI-compatible async AI provider for Ollama and Aidos."""

import base64
import json
import re
from typing import Any

import httpx

from vision_memory_book.ai.aidos_discovery import discover_aidos_endpoint
from vision_memory_book.ai.base import VisionResult
from vision_memory_book.config import Settings

DEFAULT_VISION_PROMPT = (
    "Analyze this photo and respond with a single JSON object containing:\n"
    '- "description": a concise 1-3 sentence description of the scene, lighting, and moment\n'
    '- "tags": an array of 5-15 lowercase descriptive keyword tags. Always include '
    "specific tags when visible for:\n"
    '  * documents/paperwork: "receipt", "invoice", "document", "ticket", "screenshot", "handwriting"\n'
    '  * people & roles: "child", "children", "baby", "family", "elderly", "priest", "clergy"\n'
    '  * places & ceremonies: "church", "chapel", "cemetery", "coffin", "funeral", "wedding", "baptism", "birthday"\n'
    '- "scene": a short setting/scene label (e.g. "beach sunset", "family dinner", "church ceremony")\n'
    '- "subjects": an array of main subjects or objects in the photo\n'
    "Return only valid JSON."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _parse_vision_json(raw_text: str) -> VisionResult:
    stripped = raw_text.strip()
    if not stripped:
        return VisionResult(description="", tags=[], scene=None, subjects=[], raw_json={})

    candidate = stripped
    fence_match = _FENCE_RE.match(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return VisionResult(
            description=stripped,
            tags=[],
            scene=None,
            subjects=[],
            raw_json={},
        )

    if not isinstance(parsed, dict):
        return VisionResult(
            description=stripped,
            tags=[],
            scene=None,
            subjects=[],
            raw_json={},
        )

    description = str(parsed.get("description") or "").strip() or stripped
    raw_tags = parsed.get("tags") or []
    tags = (
        [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        if isinstance(raw_tags, list)
        else []
    )
    raw_scene = parsed.get("scene")
    scene = str(raw_scene).strip() if raw_scene is not None and str(raw_scene).strip() else None
    raw_subjects = parsed.get("subjects") or []
    subjects = (
        [str(s).strip() for s in raw_subjects if str(s).strip()]
        if isinstance(raw_subjects, list)
        else []
    )

    return VisionResult(
        description=description,
        tags=tags,
        scene=scene,
        subjects=subjects,
        raw_json=parsed,
    )


class OpenAICompatProvider:
    """Async OpenAI-compatible client supporting Ollama, Aidos, and split-provider setups."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        vision_model: str = "qwen2.5vl",
        embed_model: str = "nomic-embed-text",
        text_model: str = "qwen2.5:14b",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.vision_model = vision_model
        self.embed_model = embed_model
        self.text_model = text_model
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=trust_env)
        )

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this provider."""
        if self._owns_client:
            await self._client.aclose()

    async def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str | None = None,
    ) -> VisionResult:
        """Analyze an image using an OpenAI-compatible vision chat completion."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        payload: dict[str, Any] = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.2,
        }

        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        return _parse_vision_json(_extract_message_text(raw_content))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors via POST /embeddings, sorted by index."""
        if not texts:
            return []

        payload = {
            "model": self.embed_model,
            "input": texts,
        }
        resp = await self._client.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        indexed = sorted(
            enumerate(items),
            key=lambda pair: pair[1].get("index", pair[0]),
        )
        return [[float(x) for x in item["embedding"]] for _, item in indexed]

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        """Generate a chat completion via POST /chat/completions."""
        payload = {
            "model": self.text_model,
            "messages": messages,
        }
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        return _extract_message_text(raw_content)


def build_ai_providers(settings: Settings) -> dict[str, OpenAICompatProvider]:
    """Build split-provider OpenAICompatProvider instances for vision, embed, and text."""
    uses_aidos = any(
        p.strip().lower() == "aidos"
        for p in (
            settings.vision_provider,
            settings.embed_provider,
            settings.text_provider,
        )
    )
    aidos_ep = (
        discover_aidos_endpoint(settings.aidos_discovery_path) if uses_aidos else None
    )

    def _resolve(provider_name: str, configured_url: str, configured_key: str) -> tuple[str, str | None]:
        if provider_name.strip().lower() == "aidos" and aidos_ep is not None:
            return aidos_ep.base_url, configured_key or aidos_ep.token
        return configured_url, configured_key or None

    vision_url, vision_key = _resolve(
        settings.vision_provider, settings.vision_base_url, settings.vision_api_key
    )
    embed_url, embed_key = _resolve(
        settings.embed_provider, settings.embed_base_url, settings.embed_api_key
    )
    text_url, text_key = _resolve(
        settings.text_provider, settings.text_base_url, settings.text_api_key
    )

    return {
        "vision": OpenAICompatProvider(
            base_url=vision_url,
            api_key=vision_key,
            vision_model=settings.vision_model,
            embed_model=settings.embed_model,
            text_model=settings.text_model,
        ),
        "embed": OpenAICompatProvider(
            base_url=embed_url,
            api_key=embed_key,
            vision_model=settings.vision_model,
            embed_model=settings.embed_model,
            text_model=settings.text_model,
        ),
        "text": OpenAICompatProvider(
            base_url=text_url,
            api_key=text_key,
            vision_model=settings.vision_model,
            embed_model=settings.embed_model,
            text_model=settings.text_model,
        ),
    }
