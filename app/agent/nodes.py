import itertools
import json
import re
import time
from functools import wraps
from typing import Callable, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.agent import prompts
from app.agent.state import GraphState
from app.agent.tools import web_search
from app.config import get_settings
from app.core.llm import get_generator_llm, get_grader_llm, get_router_llm
from app.core.logging import logger
from app.retrieval.retriever import retrieve


# Monotonic sequence so we can order steps even when timestamps tie.
_step_seq = itertools.count(1)


def _timed(name: str) -> Callable:
    """Decorator: measure wall-clock elapsed of a node and append to state['timings']."""

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: GraphState) -> GraphState:
            t0 = time.perf_counter()
            patch = fn(state) or {}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            seq = next(_step_seq)
            logger.info("[timing] {:>22s}  {:>8.1f} ms  (#{})", name, elapsed_ms, seq)
            existing = patch.get("timings") or []
            patch["timings"] = existing + [
                {"step": name, "elapsed_ms": round(elapsed_ms, 2), "seq": seq}
            ]
            return patch

        return wrapper

    return deco


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Some reasoning models forget the opening <think> but still emit </think>
_DANGLING_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> chain-of-thought emitted by reasoning models.

    Handles three shapes:
      1. `<think>...</think>\\nactual answer`
      2. `...thoughts...</think>\\nactual answer` (missing opening tag)
      3. plain text without think tags — returned as-is.
    """
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _DANGLING_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _parse_json(text: str) -> dict:
    """Robust JSON extractor — strips <think> blocks then extracts JSON."""
    if not text:
        return {}
    text = _strip_thinking(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _format_context(docs: List[Document]) -> str:
    if not docs:
        return "（无）"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[来源 {i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)


# ---------------- nodes ----------------


@_timed("rewrite_query")
def rewrite_query(state: GraphState) -> GraphState:
    """First-pass query cleanup before routing."""
    question = state["question"]
    chain = prompts.QUERY_REWRITE_PROMPT | get_router_llm() | StrOutputParser()
    raw = chain.invoke({"question": question})
    rewritten = _strip_thinking(raw).strip().strip('"').splitlines()[-1].strip()
    if len(rewritten) > 300:
        rewritten = rewritten[:300]
    logger.info("rewrite_query: '{}' -> '{}'", question, rewritten)
    return {
        "rewritten_question": rewritten or question,
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "timings": [],  # initialize so downstream concat is well-defined
    }


@_timed("route_question")
def route_question(state: GraphState) -> GraphState:
    """LLM decides between local vectorstore and web search."""
    q = state.get("rewritten_question") or state["question"]
    chain = prompts.ROUTER_PROMPT | get_router_llm() | StrOutputParser()
    raw = chain.invoke({"question": q})
    parsed = _parse_json(raw)
    route = parsed.get("route", "vectorstore")
    if route not in ("vectorstore", "web_search"):
        route = "vectorstore"
    logger.info("route_question -> {}", route)
    return {"route": route}


@_timed("retrieve")
def retrieve_docs(state: GraphState) -> GraphState:
    q = state.get("rewritten_question") or state["question"]
    docs = retrieve(q)
    attempts = state.get("retrieval_attempts", 0) + 1
    logger.info("retrieve_docs: {} docs (attempt {})", len(docs), attempts)
    return {"documents": docs, "retrieval_attempts": attempts}


@_timed("grade_documents")
def grade_documents(state: GraphState) -> GraphState:
    """Filter retrieved docs by LLM-based relevance check."""
    q = state.get("rewritten_question") or state["question"]
    docs = state.get("documents", [])
    if not docs:
        return {"documents": []}
    chain = prompts.DOC_GRADER_PROMPT | get_grader_llm() | StrOutputParser()
    kept: List[Document] = []
    for d in docs:
        raw = chain.invoke({"question": q, "document": d.page_content[:1500]})
        verdict = _parse_json(raw).get("relevant", "no").lower()
        if verdict == "yes":
            kept.append(d)
    logger.info("grade_documents: {}/{} relevant", len(kept), len(docs))
    return {"documents": kept}


@_timed("transform_query")
def transform_query(state: GraphState) -> GraphState:
    """Rewrite the query when retrieval came back too thin."""
    previous = state.get("rewritten_question") or state["question"]
    chain = prompts.QUERY_TRANSFORM_PROMPT | get_router_llm() | StrOutputParser()
    raw = chain.invoke({"question": state["question"], "previous": previous})
    new_q = _strip_thinking(raw).strip().strip('"').splitlines()[-1].strip()
    # Keep queries short — Tavily caps at 400 chars and vector retrieval also degrades on bloat
    if len(new_q) > 300:
        new_q = new_q[:300]
    logger.info("transform_query: '{}' -> '{}'", previous, new_q)
    return {"rewritten_question": new_q or previous}


@_timed("web_search")
def do_web_search(state: GraphState) -> GraphState:
    q = state.get("rewritten_question") or state["question"]
    docs = web_search(q)
    logger.info("do_web_search: got {} results", len(docs))
    existing = state.get("documents", []) or []
    return {"documents": existing + docs, "route": "web_search"}


@_timed("generate")
def generate(state: GraphState) -> GraphState:
    q = state["question"]
    docs = state.get("documents", [])
    context = _format_context(docs)
    chain = prompts.GENERATION_PROMPT | get_generator_llm(streaming=False) | StrOutputParser()
    answer = _strip_thinking(chain.invoke({"context": context, "question": q}))
    citations = [
        {
            "index": i + 1,
            "source": d.metadata.get("source"),
            "title": d.metadata.get("title") or d.metadata.get("filename"),
            "score": d.metadata.get("score"),
            "origin": d.metadata.get("origin", "vectorstore"),
        }
        for i, d in enumerate(docs)
    ]
    attempts = state.get("generation_attempts", 0) + 1
    logger.info("generate: {} chars (attempt {})", len(answer), attempts)
    return {"generation": answer, "citations": citations, "generation_attempts": attempts}


@_timed("hallucination_grader")
def grade_hallucination(state: GraphState) -> GraphState:
    docs = state.get("documents", [])
    context = _format_context(docs)
    chain = prompts.HALLUCINATION_GRADER_PROMPT | get_grader_llm() | StrOutputParser()
    raw = chain.invoke({"context": context, "generation": state.get("generation", "")})
    grounded = _parse_json(raw).get("grounded", "yes").lower() == "yes"
    logger.info("grade_hallucination: grounded={}", grounded)
    return {"hallucinated": not grounded}


@_timed("answer_grader")
def grade_answer(state: GraphState) -> GraphState:
    chain = prompts.ANSWER_GRADER_PROMPT | get_grader_llm() | StrOutputParser()
    raw = chain.invoke({"question": state["question"], "generation": state.get("generation", "")})
    useful = _parse_json(raw).get("useful", "yes").lower() == "yes"
    logger.info("grade_answer: useful={}", useful)
    return {"answer_relevant": useful}


# ---------------- edges (conditional) ----------------


def edge_after_route(state: GraphState) -> str:
    return "web_search" if state.get("route") == "web_search" else "retrieve"


def edge_after_grade_docs(state: GraphState) -> str:
    """After grading: generate if we have relevant docs, else rewrite or escalate."""
    settings = get_settings()
    docs = state.get("documents", [])
    if docs:
        return "generate"
    if state.get("retrieval_attempts", 0) >= settings.retrieval_max_retries:
        return "web_search"  # escalate to web after exhausting rewrites
    return "transform_query"


def edge_after_hallucination(state: GraphState) -> str:
    settings = get_settings()
    if state.get("hallucinated"):
        if state.get("generation_attempts", 0) >= settings.retrieval_max_retries:
            return "answer_grader"  # give up retrying, let the answer grader decide
        return "generate"
    return "answer_grader"


def edge_after_answer_grader(state: GraphState) -> str:
    settings = get_settings()
    if state.get("answer_relevant"):
        return "END"
    if state.get("retrieval_attempts", 0) >= settings.retrieval_max_retries + 1:
        return "END"
    return "transform_query"
