# Footnote

[![CI](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml)

**A RAG agent over arXiv papers that verifies its own citations before you see the answer.**

[Live Demo](https://ragchatbot-ui-three.vercel.app) · [Frontend Repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

---

## What Makes This Different

- **Citations are verified, not trusted.** Every arXiv ID in the answer is checked against what the tools actually returned. Fake IDs or mismatched titles get caught — after 2 failed retries, the agent refuses rather than hallucinate.
- **Tool calls are controlled by code, not the LLM.** The agent can search but cannot call `extract_info` — the pipeline invokes it deterministically once qualifying papers are found (`score ≥ 0.7`).
- **Clarification is grounded and bounded.** Ambiguous queries pause via LangGraph interrupt with options sourced from real paper titles. One clarification max — then it force-searches.
- **Cache invalidates itself.** Verified answers are cached with an MD5 fingerprint of the paper library. Add or remove a paper, and stale cache entries are gone.
- **Conversation memory stays bounded.** Tool payloads (~95% of checkpoint size) are stripped every turn. Only the last 6 exchanges survive.

---

## How It Works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> P["prune — trim history to 6 turns"]
    P --> B["check_cache — cosine ≥ 0.92 + corpus fingerprint"]
    B -- Hit --> C["END — return cached answer"]
    B -- Miss --> D["run_agent — hybrid search + LLM relevance judge
    → deterministic extract_info → synthesis"]
    D -- "Ambiguous (1st time)" --> E["clarify — LangGraph interrupt"]
    E -- "User responds" --> D
    D --> G["check_citations — verify arXiv IDs + title overlap ≥ 0.3"]
    G -- Pass --> H["END — stream answer, cache if earned"]
    G -- "Fail (retries < 2)" --> I["retry — inject specific feedback"]
    I --> D
    G -- "Fail (retries ≥ 2)" --> J["fallback — safe refusal"]
    J --> H
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + LangChain |
| Backend | FastAPI, Uvicorn, SSE streaming |
| Tool Protocol | FastMCP (in-process, mounted at `/mcp`) |
| LLM | Cerebras `gpt-oss-120b` |
| Search | Hybrid BM25 (`rank-bm25`) + dense vectors (`fastembed`, `all-MiniLM-L6-v2`, 384-dim) |
| Database | PostgreSQL + `pgvector` (IVFFlat for papers, HNSW for semantic cache) |
| External Search | arXiv API (live fallback when local search is insufficient) |
| Observability | Langfuse (tracing, session grouping, online scoring) |
| Rate Limiting | SlowAPI (10 req/min per IP) |
| Frontend | React + Vite ([separate repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)) |
| Deployment | Render (backend) · Vercel (frontend) |

---

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL with `pgvector` extension (e.g., [Neon](https://neon.tech))
- Cerebras API key

### Install

```bash
git clone https://github.com/AnuradhaBhanout/Footnote.git
cd Footnote
uv pip install -e ".[dev]"
```

### Configure

Create `.env` in the project root:

```env
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
CEREBRAS_API_KEY=csk-your-key

# Optional
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
ALLOWED_ORIGINS=http://localhost:5173
```

Tables are created automatically on first startup.

### Run

```bash
# Linux / macOS
cd src && uvicorn api.api:app --host 0.0.0.0 --port 8000 --workers 1

# Windows
cd src && python run_dev.py
```

> [!IMPORTANT]
> **Single worker only.** `HybridIndex` and `SemanticCache` are in-memory singletons — multiple workers create unsynchronized copies.

---

## Project Structure

```
Footnote/
├── pyproject.toml
├── src/
│   ├── api/
│   │   ├── api.py                  # FastAPI app, CORS, rate limits, /chat, /resume
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── sse.py                  # SSE stream generator + post-stream cache storage
│   ├── client/
│   │   ├── agent_prompt.py         # System prompt + tool exclusion list
│   │   └── mcp_v1_chatBot.py      # MCP client lifecycle + agent builder
│   ├── db/
│   │   ├── citation_verifier.py    # arXiv ID extraction + title overlap check
│   │   ├── rag_index.py            # HybridIndex: BM25 + dense vectors
│   │   └── semantic_cache.py       # Corpus-versioned cosine cache (HNSW)
│   ├── graph/
│   │   ├── graph_pipeline.py       # StateGraph compilation + edge wiring
│   │   ├── nodes.py                # All graph nodes (prune, cache, agent, verify, fallback)
│   │   ├── routing.py              # Conditional routing predicates
│   │   └── helpers.py              # Pruning logic, paper ID collection
│   ├── server/
│   │   ├── tools.py                # MCP tools: hybrid_search, search, extract_info
│   │   └── relevance.py            # LLM-as-a-judge relevance evaluator
│   ├── evals/
│   │   ├── queries.jsonl           # 20-query evaluation dataset
│   │   └── run_eval.py             # Recall@5 + MRR benchmark
│   └── tests/                      # 48 unit and regression tests
└── .github/workflows/ci.yml
```

---

## API Endpoints

| Endpoint | Method | Type | Description |
|---|---|---|---|
| `/chat` | POST | SSE stream | Send query, receive streamed answer with tool events |
| `/resume` | POST | SSE stream | Resume after clarification interrupt |
| `/feedback` | POST | JSON | Submit thumbs-up/down (logged to Langfuse) |
| `/health` | GET | JSON | Readiness check + DB connectivity |
| `/mcp/sse` | GET/POST | SSE | In-process MCP tool server |

### SSE Events

`tool_start` · `tool_end` · `token` · `interrupt` · `done` · `error`

---

## Testing

```bash
uv run --extra dev --directory src pytest -q
```

48 tests covering: citation verification, routing predicates, agent recovery (5-branch cascade), hybrid search, pruning logic, and eval metrics.

---

## Observability

Every request is traced in Langfuse with deterministic online scores:

| Score | When | Meaning |
|---|---|---|
| `cache_hit` | Every turn | `1` = answered from cache |
| `citation_pass_rate` | Only when citations exist | `1` = all citations verified |
| `error` | On failure | `1` = execution failed or degraded |
| `user_feedback` | User action | `1` = thumbs up, `0` = thumbs down |

---

## Deployment

Deployed as a single service on **Render** (free tier).

**Build:**
```bash
pip install -r src/requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"
```

**Start:**
```bash
cd src && uvicorn api.api:app --host 0.0.0.0 --port $PORT --workers 1
```

> [!NOTE]
> Free-tier Render spins down after 15 min of inactivity. First request triggers a cold start. Hit `/health` periodically to keep it warm.

---

## Key Constants

| Constant | Default | What it controls |
|---|---|---|
| `SCORE_FLOOR` | `0.7` | Min hybrid score for deterministic extraction |
| `SIMILARITY_THRESHOLD` | `0.92` | Min cosine similarity for cache hit |
| `MAX_RETRIES` | `2` | Retry cap for search + citation failures |
| `keep_turns` | `6` | Conversation turns retained in memory |
| `overlap_threshold` | `0.3` | Min title-word overlap for citation pass |
| Rate limit | `10/min` | Per-IP POST request cap |

---


