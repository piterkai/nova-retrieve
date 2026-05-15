# Nova Retrieve — 企业级 Agentic RAG

基于 **LangChain + LangGraph + Qdrant + BGE-M3** 的可生产部署 Agentic RAG 框架。

> 英文版：[README.md](README.md)

## 特性

- **LangGraph 状态机驱动**：查询改写 → 路由 → 检索 → 文档评分 → 生成 → 幻觉检测 → 答案有用性评估，全部带回退路径。
- **CRAG / Self-RAG 思路**：检索不够时自动改写重试；生成有幻觉时自动重写；多轮失败兜底走 Web 搜索。
- **本地向量化（BGE-M3）**：通过 sentence-transformers 加载本地模型目录，数据不出域。设 `EMBEDDING_LOCAL_PATH` 即可。
- **Qdrant 向量库**：Docker 一键起,自动建集合。
- **OpenAI 兼容 LLM**：可对接 DeepSeek / 通义 / 智谱 / 任何兼容 endpoint。
- **Tavily Web 兜底**：知识库召回不足时切换实时网络搜索。
- **FastAPI + SSE 流式**：节点级事件流，前端可实时展示 agent 推理轨迹。
- **内置 Web UI**：零构建的单页前端（`/ui/`），实时显示每个 agent 步骤、耗时、引用来源。

## 架构

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

## 快速开始

### 1. 起 Qdrant

```bash
docker compose up -d qdrant
```

### 2. 装依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

把已下载的 BGE-M3 目录路径填入 `.env` 的 `EMBEDDING_LOCAL_PATH`（应包含 `config.json` / `tokenizer.json` / `model.safetensors` 等）。若留空则首次运行从 HuggingFace 下载。

### 3. 配置

```bash
cp .env.example .env
# 编辑 LLM_BASE_URL / LLM_API_KEY / TAVILY_API_KEY
```

### 4. 灌库

```bash
python -m scripts.ingest_docs ./data/docs
```

### 5. 启服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 <http://localhost:8000/> 即进入 Web UI（自动重定向到 `/ui/`）。

或交互式 CLI：

```bash
python -m scripts.chat_cli
```

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/ingest` | POST | 摄入文件/目录 |
| `/chat` | POST | 阻塞式问答，返回完整答案 + 引用 |
| `/chat/stream` | POST | SSE 流：`step` 事件按节点推送，`answer` 事件返回最终结果 |

### 示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"我们的退款政策是什么？"}'
```

SSE 流：

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"GPT-5 发布了吗？"}'
```

## 目录结构

```
app/
├── config.py           # pydantic-settings 配置
├── main.py             # FastAPI app + /ui 静态挂载
├── api/                # 路由与 schema
├── core/               # llm / embeddings / vectorstore / logging
├── ingest/             # loader / chunker / pipeline
├── retrieval/          # 检索器
└── agent/              # state / nodes / edges / prompts / graph / tools
web/                    # 前端单页（无构建）
├── index.html
├── styles.css
└── app.js
scripts/
├── ingest_docs.py
└── chat_cli.py
```

## 可扩展点

- **Reranker**：在 `retrieval/` 加 BGE-Reranker 二阶段精排。
- **多租户**：`ChatRequest.collection` 已经留好按租户隔离的钩子。
- **缓存**：在 `route_question` / `grade_documents` 上加 Redis 缓存可显著降本。
- **观测**：设置 `LANGSMITH_API_KEY` 即可全链路追踪。
