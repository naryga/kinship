"""Runtime configuration, read from environment variables (and `.env` if present)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Matches docker-compose.yml and .env.example: a local development database, not a secret.
LOCAL_DATABASE_URL = "postgresql+psycopg://kinship:kinship@localhost:5432/kinship"


class Settings(BaseSettings):
    """Kinship settings. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = LOCAL_DATABASE_URL
    openai_api_key: str = ""
    # `openai` or `fake`; empty means `openai`. See `kinship.rag.embedder.get_embedder`.
    kinship_embedder: str = ""
    # v0 auth: comma-separated `token:owner_id` pairs.
    kinship_tokens: str = ""

    @property
    def token_owner_ids(self) -> dict[str, int]:
        """`KINSHIP_TOKENS` parsed into a token -> owner_id mapping."""
        pairs: dict[str, int] = {}
        for item in self.kinship_tokens.split(","):
            if not item.strip():
                continue
            token, _, owner_id = item.strip().rpartition(":")
            if not token:
                raise ValueError("KINSHIP_TOKENS entries must look like 'token:owner_id'")
            pairs[token] = int(owner_id)
        return pairs


@lru_cache
def get_settings() -> Settings:
    return Settings()
