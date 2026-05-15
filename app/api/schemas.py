from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: Optional[str] = None


class Citation(BaseModel):
    index: int
    source: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    origin: Optional[str] = None


class StepTiming(BaseModel):
    step: str
    elapsed_ms: float
    seq: int


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    route: Optional[str] = None
    hallucinated: Optional[bool] = None
    answer_relevant: Optional[bool] = None
    timings: List[StepTiming] = []
    total_ms: float = 0.0


class IngestRequest(BaseModel):
    paths: List[str] = Field(..., min_length=1)
    collection: Optional[str] = None
    chunk_size: int = 800
    chunk_overlap: int = 120


class IngestResponse(BaseModel):
    loaded: int
    chunks: int
    collection: str
