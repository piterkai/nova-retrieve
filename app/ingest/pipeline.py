from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from app.core.logging import logger
from app.core.vectorstore import get_vectorstore
from app.ingest.chunker import split_documents
from app.ingest.loaders import load_path


def ingest_paths(
    paths: Iterable[str | Path],
    collection: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    batch_size: int = 64,
) -> dict:
    """End-to-end ingestion: load → chunk → embed → upsert into Qdrant."""
    all_docs: List[Document] = []
    for p in paths:
        all_docs.extend(load_path(p))
    if not all_docs:
        logger.warning("No documents loaded.")
        return {"loaded": 0, "chunks": 0}

    chunks = split_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Split into {} chunks. Indexing into Qdrant...", len(chunks))

    store = get_vectorstore(collection)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        store.add_documents(batch)
        logger.info("Indexed batch {}-{} ({} chunks)", i, i + len(batch), len(batch))

    return {"loaded": len(all_docs), "chunks": len(chunks)}
