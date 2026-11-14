"""Abstract AI provider contracts for vision, embeddings, and text generation."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    """Structured output from a Vision Language Model for a single photo."""

    description: str
    tags: list[str] = Field(default_factory=list)
    scene: str | None = None
    subjects: list[str] = Field(default_factory=list)
    raw_json: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class AIProvider(Protocol):
    """Async protocol for local AI providers (Ollama, Aidos, OpenAI-compatible)."""

    async def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str | None = None,
    ) -> VisionResult:
        """Analyze an image and return a structured VisionResult."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""
        ...

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        """Generate a chat completion response for OpenAI-format messages."""
        ...
