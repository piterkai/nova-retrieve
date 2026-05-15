from typing import List

from langchain_core.documents import Document

from app.config import get_settings
from app.core.logging import logger


TAVILY_MAX_QUERY_CHARS = 380  # API limit is 400; leave headroom


def web_search(query: str) -> List[Document]:
    """Tavily-powered web search returning LangChain Documents.

    Returns an empty list if TAVILY_API_KEY is missing — callers should fall
    back to generating from whatever they have rather than failing.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — web search disabled, returning empty.")
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.error("tavily-python not installed.")
        return []

    q = (query or "").strip()
    if len(q) > TAVILY_MAX_QUERY_CHARS:
        logger.warning("web_search query truncated from {} to {} chars", len(q), TAVILY_MAX_QUERY_CHARS)
        q = q[:TAVILY_MAX_QUERY_CHARS]

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        resp = client.search(
            query=q,
            max_results=settings.tavily_max_results,
            search_depth="advanced",
            include_answer=False,
        )
    except Exception as e:
        logger.error("Tavily search failed: {}", e)
        return []
    docs: List[Document] = []
    for r in resp.get("results", []):
        content = r.get("content") or ""
        url = r.get("url", "")
        title = r.get("title", "")
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "origin": "web_search",
                    "score": r.get("score", 0.0),
                },
            )
        )
    return docs
