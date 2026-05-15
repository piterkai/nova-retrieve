from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache(maxsize=4)
def get_chat_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at any OpenAI-compatible endpoint.

    Caching keys on (model, temperature, streaming) so the same client is reused
    when callers request the same configuration.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
        streaming=streaming,
        max_retries=2,
    )


def get_router_llm() -> ChatOpenAI:
    settings = get_settings()
    return get_chat_llm(model=settings.llm_router_model, temperature=0.0)


def get_grader_llm() -> ChatOpenAI:
    return get_chat_llm(temperature=0.0)


def get_generator_llm(streaming: bool = True) -> ChatOpenAI:
    return get_chat_llm(streaming=streaming)
