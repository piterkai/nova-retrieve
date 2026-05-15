from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.core.logging import logger, setup_logging
from app.core.vectorstore import ensure_collection


# Project root -> ./web (frontend single-page app, no build step).
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info("Starting Nova Retrieve (model={}, qdrant={})", settings.llm_model, settings.qdrant_url)
    # Warm up — fail fast on misconfig.
    get_embeddings()
    ensure_collection()
    yield
    logger.info("Shutting down Nova Retrieve")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Nova Retrieve — Agentic RAG",
        description="Enterprise Agentic RAG with LangChain + LangGraph + Qdrant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # Frontend: serve /web as /ui/, redirect root for convenience.
    if WEB_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def _root():
            return RedirectResponse(url="/ui/")
    else:
        logger.warning("Frontend directory not found at {}, /ui disabled", WEB_DIR)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.app_host, port=s.app_port, reload=False)
