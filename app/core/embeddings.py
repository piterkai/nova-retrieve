from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load BGE-M3 (or any sentence-transformers model) from a local path.

    Priority:
        1. EMBEDDING_LOCAL_PATH — load from local directory (no network).
        2. EMBEDDING_MODEL      — HuggingFace repo id, downloaded on first use.
    """
    settings = get_settings()
    local = settings.embedding_local_path
    if local:
        p = Path(local).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"EMBEDDING_LOCAL_PATH not found: {p}")
        model_id = str(p)
        logger.info("Loading embedding model from local path: {}", model_id)
    else:
        model_id = settings.embedding_model
        logger.info("Loading embedding model by repo id: {}", model_id)

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
