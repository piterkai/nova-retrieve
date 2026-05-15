import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_graph
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    IngestRequest,
    IngestResponse,
    StepTiming,
)
from app.config import get_settings
from app.core.logging import logger
from app.ingest.pipeline import ingest_paths


router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    settings = get_settings()
    try:
        result = await asyncio.to_thread(
            ingest_paths,
            req.paths,
            req.collection,
            req.chunk_size,
            req.chunk_overlap,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(
        loaded=result["loaded"],
        chunks=result["chunks"],
        collection=req.collection or settings.qdrant_collection,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Blocking endpoint — returns the full answer after the graph terminates."""
    graph = get_graph()
    t0 = time.perf_counter()
    final_state = await asyncio.to_thread(graph.invoke, {"question": req.question})
    total_ms = (time.perf_counter() - t0) * 1000.0
    timings_raw = final_state.get("timings") or []
    timings = [StepTiming(**t) for t in timings_raw]
    summary = ", ".join(f"{t.step}={t.elapsed_ms:.0f}ms" for t in timings)
    logger.info("[timing] total={:.0f} ms | {}", total_ms, summary)
    return ChatResponse(
        answer=final_state.get("generation", ""),
        citations=[Citation(**c) for c in final_state.get("citations", [])],
        route=final_state.get("route"),
        hallucinated=final_state.get("hallucinated"),
        answer_relevant=final_state.get("answer_relevant"),
        timings=timings,
        total_ms=round(total_ms, 2),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint streaming node-level events and the final answer.

    Event types:
        - step:    a node finished. data = {"node": str, "state_delta": {...}}
        - answer:  final answer ready. data = {"answer": str, "citations": [...]}
        - error:   exception. data = {"detail": str}
    """
    graph = get_graph()

    async def event_stream() -> AsyncGenerator[dict, None]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def run_graph():
            try:
                t0 = time.perf_counter()
                final_state = {}
                accumulated_timings = []
                for chunk in graph.stream({"question": req.question}, stream_mode="updates"):
                    # chunk is {node_name: {state fields...}}
                    for node_name, state_delta in chunk.items():
                        if isinstance(state_delta, dict):
                            final_state.update(state_delta)
                            for t in state_delta.get("timings") or []:
                                accumulated_timings.append(t)
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("step", {"node": node_name, "state_delta": _safe(state_delta)}),
                        )
                total_ms = (time.perf_counter() - t0) * 1000.0
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    (
                        "answer",
                        {
                            "answer": final_state.get("generation", ""),
                            "citations": final_state.get("citations", []),
                            "route": final_state.get("route"),
                            "hallucinated": final_state.get("hallucinated"),
                            "answer_relevant": final_state.get("answer_relevant"),
                            "timings": accumulated_timings,
                            "total_ms": round(total_ms, 2),
                        },
                    ),
                )
            except Exception as e:
                logger.exception("Graph stream failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", {"detail": str(e)}))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        asyncio.create_task(asyncio.to_thread(run_graph))

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            event, data = item
            yield {"event": event, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(event_stream())


def _safe(obj):
    """Convert non-JSON-serializable bits (LangChain Documents) to dicts."""
    from langchain_core.documents import Document

    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(x) for x in obj]
    if isinstance(obj, Document):
        return {"page_content": obj.page_content[:300], "metadata": obj.metadata}
    return obj
