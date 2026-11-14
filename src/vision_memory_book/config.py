"""Application configuration via Pydantic BaseSettings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for Vision Memory Book."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("./data")
    db_path: Path = Path("data/vmb.db")
    cache_dir: Path = Path("data/cache")
    originals_dir: Path = Path("data/originals")

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_token_path: Path = Path("data/token.json")

    # Download
    cache_max_dim: int = 1024
    download_concurrency: int = 4

    # AI — Vision
    vision_provider: str = "ollama"
    vision_base_url: str = "http://localhost:11434/v1"
    vision_api_key: str = ""
    vision_model: str = "qwen2.5vl"

    # AI — Embeddings
    embed_provider: str = "ollama"
    embed_base_url: str = "http://localhost:11434/v1"
    embed_api_key: str = ""
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # AI — Text LLM
    text_provider: str = "ollama"
    text_base_url: str = "http://localhost:11434/v1"
    text_api_key: str = ""
    text_model: str = "qwen2.5:14b"

    # Aidos RFC-0103 discovery
    aidos_discovery_path: str | None = None

    def ensure_dirs(self) -> None:
        """Ensure required local data directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.google_token_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application Settings instance."""
    return Settings()
