from typing import List, Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vectorstore import get_vectorstore


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    collection: Optional[str] = None,
) -> List[Document]:
    """Semantic retrieval with score filtering.

    Returns documents whose similarity score >= threshold. The score is attached
    to ``metadata["score"]`` so downstream graders can use it.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    threshold = settings.retrieval_score_threshold if score_threshold is None else score_threshold
    store = get_vectorstore(collection)
    pairs = store.similarity_search_with_score(query, k=k)
    docs: List[Document] = []
    for doc, score in pairs:
        if score >= threshold:
            doc.metadata["score"] = float(score)
            docs.append(doc)
    return docs
