# Nova Retrieve — Enterprise Agentic RAG

A production-ready Agentic RAG framework built on **LangChain + LangGraph + Qdrant + BGE-M3**.

> 中文版：[README.zh.md](README.zh.md)

## Features

- **LangGraph state machine** — query rewriting → routing → retrieval → document grading → generation → hallucination check → answer-usefulness check, every edge has a fallback.
- **CRAG / Self-RAG inspired** — auto-rewrites the query when retrieval is thin, auto-regenerates when the answer is hallucinated, and falls back to web search after exhausting retries.
- **Local embeddings (BGE-M3)** — loads the model from a local directory via `sentence-transformers`; your data never leaves your network. Just set `EMBEDDING_LOCAL_PATH`.
- **Qdrant vector store** — single-command Docker deployment, collection is auto-created on first run.
- **OpenAI-compatible LLM** — plug in DeepSeek / Qwen / Zhipu / any compatible endpoint.
- **Tavily web fallback** — switches to live web search when the local index can't answer.
- **FastAPI + SSE streaming** — per-node event stream that lets the frontend render the agent's reasoning trace live.
- **Built-in Web UI** — zero-build single-page frontend served at `/ui/`, with live agent step / timing / citation rendering.

## Architecture

```
                ┌──────────────┐
                │ rewrite_query│
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │route_question│
                └──┬────────┬──┘
       vectorstore│        │web_search
                  ▼        ▼
              ┌──────┐  ┌─────────┐
              │retrieve  │web_search│
              └──┬───┘  └────┬────┘
                 ▼           │
        ┌──────────────┐    │
        │grade_documents│    │
        └──┬───────┬────┘    │
   relevant│  none │         │
           ▼       ▼         │
       ┌────────┐ transform  │
       │generate│◄──query──┐ │
       └───┬────┘          │ │
           ▼          retry│ │
   ┌────────────────┐      │ │
   │hallucination_  │──no──┘ │
   │grader (CRAG)   │        │
   └───┬────────────┘        │
       ▼ yes                 │
   ┌────────────┐            │
   │answer_grader│──no→transform_query
   └───┬────────┘
       ▼ useful
      END
```

## Quick start

### 1. Start Qdrant

```bash
docker compose up -d qdrant
```

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Point `EMBEDDING_LOCAL_PATH` in `.env` at your local BGE-M3 directory (it should contain `config.json`, `tokenizer.json`, `model.safetensors`, etc.). Leave it empty to pull the model from HuggingFace on first run.

### 3. Configure

```bash
cp .env.example .env
# edit LLM_BASE_URL / LLM_API_KEY / TAVILY_API_KEY
```

### 4. Ingest documents

```bash
python -m scripts.ingest_docs ./data/docs
```

### 5. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/> in a browser — you'll be redirected to the Web UI at `/ui/`.

Or use the interactive CLI:

```bash
python -m scripts.chat_cli
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest files or directories |
| `/chat` | POST | Blocking endpoint — returns the full answer with citations |
| `/chat/stream` | POST | SSE stream — `step` events per node, final `answer` event with the result |

### Examples

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is our refund policy?"}'
```

SSE stream:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Has GPT-5 been released?"}'
```

## Project layout

```
app/
├── config.py           # pydantic-settings config
├── main.py             # FastAPI app + /ui static mount
├── api/                # routes and schemas
├── core/               # llm / embeddings / vectorstore / logging
├── ingest/             # loader / chunker / pipeline
├── retrieval/          # retriever
└── agent/              # state / nodes / edges / prompts / graph / tools
web/                    # single-page frontend (no build step)
├── index.html
├── styles.css
└── app.js
scripts/
├── ingest_docs.py
└── chat_cli.py
```

## Extension points

- **Reranker** — drop a BGE-Reranker into `retrieval/` for a two-stage rerank.
- **Multi-tenancy** — `ChatRequest.collection` is already wired for per-tenant collection isolation.
- **Caching** — adding Redis caching on `route_question` / `grade_documents` cuts cost significantly.
- **Observability** — set `LANGSMITH_API_KEY` for end-to-end tracing.
