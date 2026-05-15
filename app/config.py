from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_base_url: str = Field(default="https://api.deepseek.com/v1")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="deepseek-chat")
    llm_router_model: str = Field(default="deepseek-chat")
    llm_temperature: float = Field(default=0.1)
    llm_timeout: int = Field(default=60)

    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_local_path: str = Field(default="")
    embedding_device: str = Field(default="cpu")
    embedding_dim: int = Field(default=1024)

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="nova_retrieve_docs")

    retrieval_top_k: int = Field(default=6)
    retrieval_score_threshold: float = Field(default=0.3)
    retrieval_max_retries: int = Field(default=2)

    tavily_api_key: str = Field(default="")
    tavily_max_results: int = Field(default=5)

    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_log_level: str = Field(default="INFO")
    app_cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["*"])

    @field_validator("app_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
