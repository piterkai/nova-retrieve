import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """Shared state across all LangGraph nodes.

    Only ``question`` is required on entry. Other fields are populated by nodes
    as the graph progresses.
    """

    question: str
    rewritten_question: str
    route: Literal["vectorstore", "web_search"]
    documents: List[Document]
    generation: str

    retrieval_attempts: int
    generation_attempts: int

    hallucinated: bool
    answer_relevant: bool

    citations: List[dict]
    error: Optional[str]

    # Per-node timing records, accumulated across the graph run.
    # Each entry: {"step": str, "elapsed_ms": float, "seq": int}
    timings: Annotated[List[dict], operator.add]
