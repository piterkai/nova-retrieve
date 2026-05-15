from functools import lru_cache
from typing import Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        prefer_grpc=False,
        timeout=30,
    )


def ensure_collection(collection: Optional[str] = None) -> str:
    """Create the collection lazily if it does not exist."""
    settings = get_settings()
    client = get_qdrant_client()
    name = collection or settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        logger.info("Creating Qdrant collection '{}' (dim={})", name, settings.embedding_dim)
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(
                size=settings.embedding_dim,
                distance=rest.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=name,
            field_name="metadata.source",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
    return name


def get_vectorstore(collection: Optional[str] = None) -> QdrantVectorStore:
    settings = get_settings()
    name = ensure_collection(collection or settings.qdrant_collection)
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=name,
        embedding=get_embeddings(),
    )
