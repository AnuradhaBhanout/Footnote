# Delve

**Delve** (formerly *RAGchatbot*) is a production-grade AI research assistant for researchers, writers, and bloggers. It searches, retrieves, and summarizes academic papers with verified citations — built as a real multi-user deliverable, not a demo.

## Architecture

```
Frontend (React/Vite PWA)
        │  SSE stream
        ▼
api.py (FastAPI)
        │
        ▼
graph_pipeline.py (LangGraph)
  check_cache → triage_query → run_agent → check_citations
       → retry_with_feedback / fallback → finalize
        │
        ▼
mcp_v1_chatBot.py (agent orchestrator, MCP client)
        │  MCP protocol
        ▼
research_server.py (FastMCP tool server)
  hybrid_search_papers · extract_info · search_papers
  get_available_folders · get_topic_papers
  check_semantic_cache · store_semantic_cache
        │
        ▼
PostgreSQL + pgvector
```

Citation integrity is enforced by `citation_verifier.py`, which checks every paper ID and title cited in a generated answer against what the tools actually returned — no invented papers, no invented findings.

## Stack

| Layer | Tech |
|---|---|
| Orchestration | LangGraph, LangChain |
| Tool protocol | MCP (Model Context Protocol) / FastMCP |
| Backend API | FastAPI, SSE streaming (`ReadableStream`, not `EventSource`) |
| Retrieval | Hybrid search — BM25 (`rank-bm25`) + dense embeddings (`fastembed`, ONNX) |
| Storage | PostgreSQL + `pgvector` |
| LLMs | NVIDIA `integrate.api.nvidia.com` (primary agent), Groq (evaluator) |
| Frontend | React + Vite, `react-markdown` + `remark-gfm` |
| Deployment | Render (API + research server), Vercel (frontend) |

## Key features

- **Hybrid retrieval** — combines keyword (BM25) and semantic (vector) search over your saved paper library, with an LLM-as-judge relevance check on results.
- **Citation verification** — every cited paper ID and title is checked against actual tool output before an answer is finalized; unverifiable claims trigger a retry with corrective feedback, then a safe fallback.
- **Semantic caching** — repeated or similar questions are served from a Postgres-backed semantic cache (only reliable, verified answers are ever cached — failed/fallback answers are never written).
- **Deterministic tool orchestration** — paper detail fetching (`extract_info`) is batched and driven by the graph itself rather than left to model discretion, avoiding fan-out and rate-limit exhaustion.
- **Resilient agent loop** — bounded retries, recursion limits, and reconnect handling for MCP session drops and upstream API instability.
- **Human-in-the-loop clarification** — ambiguous queries pause the graph and ask the user to disambiguate before searching.

## Project structure

```
api.py                 FastAPI backend, SSE endpoints (/chat, /resume)
graph_pipeline.py       LangGraph state machine
mcp_v1_chatBot.py       MCP client + agent orchestrator
research_server.py      FastMCP tool server (paper search/retrieval)
rag_index.py            HybridIndex — BM25 + pgvector search
semantic_cache.py       Postgres-backed semantic cache
citation_verifier.py    Post-hoc citation/fabrication check
db.py                   Shared Postgres connection + schema init
frontend/
  src/hooks/useChat.js      Chat state + SSE event handling
  src/lib/streamChat.js     SSE stream parser
```

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL with the `pgvector` extension available
- Node.js 18+ (for the frontend)
- API keys: NVIDIA (`integrate.api.nvidia.com`), Groq

### Backend

```bash
git clone <repo-url>
cd delve

# using uv
uv venv
uv pip install -r requirements.txt
```

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
NVIDIA_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run the MCP research server:

```bash
python research_server.py
```

Run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Tables are created automatically on first run via `db.init_db()`.

### Frontend

```bash
cd frontend
npm install
```

Create `.env`:

```env
VITE_API_URL=http://localhost:8000
```

```bash
npm run dev
```

## Deployment

- `research_server.py` → Render (standalone SSE service)
- `api.py` → Render (FastAPI backend)
- PostgreSQL → Render managed database
- Frontend → Vercel

Free-tier Render services spin down after ~15 minutes of inactivity; pair with an uptime monitor (e.g. UptimeRobot, 5-minute interval) if cold starts are unacceptable for your use case.

## License

Add a license before making this public — MIT is a common default for projects like this.