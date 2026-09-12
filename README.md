# Footnote
[![CI](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml)

**Ask about a topic or technique. Get one synthesized answer pulled from real papers — not a list of abstracts to read yourself.**

[Live demo](https://ragchatbot-ui-three.vercel.app) · [Frontend repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

## Why this exists

arXiv's search is literal keyword matching, not semantic. Query "recent papers on RAG" and you get zero results — that exact phrase never appears in an abstract, so you need to already know the right query syntax (just "RAG," no filler words). And even with a good query, arXiv doesn't judge whether a result actually *answers* your question, or combine matches into one answer — you do that part yourself.

Footnote closes both gaps: dense embeddings match by meaning, so natural-language queries just work, and hybrid search + an LLM relevance judge turn matches into one synthesized answer — with follow-up memory and instant caching on repeat questions.

**Same query, side by side:**

| arXiv | Footnote |
|---|---|
| ![arXiv: zero results](docs/screenshots/arxiv-no-results.png) | ![Footnote: synthesized, sourced answer](docs/screenshots/footnote-answer.png) |
| "What's new in retrieval-augmented generation?" → 0 results | Same query → synthesized summary table, every row backed by a verified citation in the sidebar |

**Scope, stated honestly:** this is a fast multi-paper *topic survey* tool, not a single-paper deep-reader. For "explain this one paper in depth," arXiv itself or a NotebookLM-style tool already does that better — Footnote doesn't ingest full PDF text.

## What it does

- **Topic → synthesized answer.** Not 10 links to click through — one answer built from the papers that actually matched.
- **Follow-ups work.** Bounded conversation memory across turns (see [state growth](#state-stayed-flat-under-load)).
- **Repeat questions are instant.** Semantic cache, invalidated automatically when the paper library changes.
- **Every citation is checked before you see it.** Invented IDs or mismatched titles get caught and retried — not shipped.

## Reliability, not just retrieval

Most agent demos hand you whatever the model produced. This one verifies it first.

| Guarantee | How |
|---|---|
| No invented citations | Every cited ID/title checked against real tool output; 2 failed retries → honest *"not enough verified info,"* never a guess |
| No runaway state | `prune` strips tool payloads + old turns every request |
| No stale cache | Cache key is fingerprinted to the current paper library — corpus changes invalidate it automatically |
| No silent failure loops | Recursion limits, retry caps, a one-clarification ceiling — enforced in code, not left to prompt instructions |

## How it works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> P["prune"]
    P --> B["check_cache"]
    B -- hit --> C[return cached answer]
    B -- miss --> D["run_agent<br/>hybrid_search_papers, search_papers, ask_clarification"]
    D -- ambiguous, 1st time --> E["clarify (interrupt)"]
    D -- ambiguous again --> K["_force_search"]
    D -- paper_ids found --> F["extract_info<br/>(deterministic, one call)"]
    E -- user answers --> D
    K --> F
    F --> G[check_citations]
    G -- passed --> H[END — cached]
    G -- "failed, retries < 2" --> I[retry_with_feedback]
    G -- "failed, retries = 2" --> J["fallback: not enough verified info"]
    I --> D
```

`hybrid_search_papers` = BM25 + dense embeddings (`fastembed`), then a separate LLM call judges relevance — not just keyword overlap. Falls back to a live arXiv search only if the library search comes back empty.

Seven nodes: `prune`, `check_cache`, `run_agent`, `clarify`, `check_citations`, `retry_with_feedback`, `fallback`.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite (separate repo), Vercel |
| Backend API | FastAPI, SSE streaming |
| Orchestration | LangGraph, LangChain (`create_agent`) |
| Tool protocol | MCP / FastMCP, mounted in-process |
| Retrieval | BM25 (`rank-bm25`) + dense embeddings (`fastembed`, ONNX) |
| Storage | PostgreSQL + `pgvector` (Neon) |
| LLM | Cerebras `gpt-oss-120b` — agent + relevance judge |
| Rate limiting | `slowapi`, 10 req/min per IP |
| Tracing | Langfuse |
| CI/CD | GitHub Actions, branch-protected `main` |
| Deployment | Render, single web service |

## Quickstart

**Prerequisites:** Python 3.12+, Postgres with `pgvector`, a Cerebras API key (Langfuse optional).

```bash
git clone https://github.com/AnuradhaBhanout/Footnote.git
cd Footnote
uv pip install -e .
```

`.env`:
```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
CEREBRAS_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
```

```bash
cd src
uvicorn api.api:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Windows:** use `python run_dev.py` instead — uvicorn hardcodes `ProactorEventLoop`, but psycopg3's async pool needs a selector loop.
>
> **Always `--workers 1`.** `HybridIndex` and `SemanticCache` are process-global singletons; a second worker = a second copy of the index and no shared state.

| Endpoint | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| MCP SSE | http://localhost:8000/mcp/sse |

Point the [frontend](https://github.com/AnuradhaBhanout/RAGchatbot-ui)'s `VITE_API_URL` at the API address above.

## Project structure

```
src/
├── api/
│   ├── api.py                  # FastAPI app: /chat, /resume, /feedback, /health; mounts MCP at /mcp
│   ├── schemas.py               # ChatRequest, ResumeRequest
│   ├── dependencies.py          # get_chatbot(), 503s until the graph is ready
│   └── sse.py                   # SSE event loop; also stores verified answers in the cache
├── client/
│   ├── mcp_v1_chatBot.py         # MCP client, connection lifecycle, agent rebuild on reconnect
│   ├── agent_prompt.py           # system prompt, EXCLUDED_FROM_AGENT tool filter
│   ├── tools.py                  # the ask_clarification tool
│   └── mcp_content.py            # normalizes MCP's content-block shapes to plain dicts
├── graph/
│   ├── graph_pipeline.py         # wiring only, builds the StateGraph
│   ├── nodes.py                  # prune, check_cache, run_agent, clarify, check_citations,
│   │                              #   retry_with_feedback, fallback, _force_search
│   ├── routing.py                # conditional edges: retry / clarify / fallback logic
│   ├── state.py                  # GraphState TypedDict
│   └── helpers.py                # stale_message_ids, _current_turn_messages,
│                                  #   paper-id collection, SCORE_FLOOR
├── server/
│   ├── mcp_app.py                # shared FastMCP instance, /health route
│   ├── tools.py                  # hybrid_search_papers, search_papers, extract_info, cache tools
│   ├── relevance.py              # the LLM-as-judge relevance evaluator
│   └── index_state.py            # HybridIndex + SemanticCache singletons, background embed
├── db/
│   ├── db.py                     # connection pool, schema init
│   ├── rag_index.py              # HybridIndex: BM25 + dense search
│   ├── embedding_model.py        # fastembed wrapper, model loaded lazily on first use
│   ├── paper_store.py            # load_all_papers, corpus fingerprint
│   ├── semantic_cache.py         # Postgres-backed answer cache, SIMILARITY_THRESHOLD
│   └── citation_verifier.py      # matches cited IDs/titles against real tool output
├── evals/
│   ├── queries.jsonl              # frozen 20-query known-item eval set
│   └── run_eval.py                # recall@5 / MRR, alpha sweep
├── run_dev.py                     # Windows-only local launcher (selector event loop)
└── tests/                         # pytest suite
```

## API

All POST routes: 10 req/min per IP. `/chat` and `/resume` stream SSE, not single JSON responses.

**`POST /chat`**
```json
{ "query": "Find work on citation faithfulness", "session_id": "optional" }
```

**`POST /resume`** — answers a pending clarification (404 if none paused)
```json
{ "session_id": "the-paused-session", "answer": "Temperature as a sampling hyperparameter" }
```

| Event | Payload | When |
|---|---|---|
| `tool_start` / `tool_end` | `{tool, input/output}` | agent calls / a tool returns |
| `interrupt` | `{question, options, session_id}` | graph pauses for clarification |
| `token` | `{content}` | streaming response token |
| `done` | `{answer, session_id, cited_paper_ids, fetched_papers, trace_id}` | graph finished |
| `error` | `{message}` | anything raised mid-stream |

**`POST /feedback`** — attaches thumbs up/down to a trace by `trace_id` from that turn's `done` event.

## Retrieval quality

Alpha sweep on a frozen 20-query known-item eval (`evals/run_eval.py`):

| alpha | 0.0 (BM25) | 0.25 | 0.5 | 0.75 | 1.0 (dense) |
|---|---|---|---|---|---|
| recall@5 | 0.850 | 0.950 | **1.000** | **1.000** | 0.950 |
| MRR | 0.677 | 0.756 | **0.883** | 0.852 | 0.842 |

Hybrid (`alpha=0.5`) beats either component alone — that's the production default. *Caveat: n=20, known-item only; doesn't measure broad topical queries or corpus-absent papers.*

## State stayed flat under load

One thread grew to 226 messages / ~400KB before `prune` was added — every node write versioned a fresh blob, nothing ever removed old turns.

| Checkpoint | `messages` size |
|---|---|
| Before fix | 409,798 bytes |
| After fix | 20,650 bytes |
| 3 turns later | 22,052 bytes |

95% reduction, flat afterward. Fix: strip tool payloads/old turns in `prune` (`db/helpers.py: stale_message_ids`), stop full-state spreads in node returns, rotate session ID on "New chat."

## Observability & evaluation

**Tracing:** one Langfuse span per request (`api/sse.py`), propagated with the session ID so turns group into Sessions. A LangChain callback handler on the graph config captures every LLM/tool call underneath — no per-node instrumentation needed.

**Online scores** — written per request, against live traffic, deterministic (not sampled judges):

| Score | Where | Meaning |
|---|---|---|
| `cache_hit` | `check_cache` | 1 on a semantic cache hit |
| `error` | `run_agent` | 1 when the turn degraded to a fallback |
| `citation_pass_rate` | `check_citations` | 1 when every citation verified — only scored when there was something to verify |
| `user_feedback` | `POST /feedback` | 1/0 from thumbs up/down in the UI |

**Offline eval:** frozen 20-query retrieval set, run on demand — see [Retrieval quality](#retrieval-quality).

**Known gap:** nothing yet judges whether a *verified* answer actually answers the question. `citation_pass_rate` proves the cited papers are real and were retrieved — not that they're relevant. Closing that needs an LLM judge over a sample of traces.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) — two required jobs on every PR:

| Job | Checks |
|---|---|
| `test` | installs pinned `src/requirements.txt`, runs pytest |
| `install-from-pyproject` | clean `pip install -e .` + import check — catches drift between `requirements.txt` and `pyproject.toml` |

Branch protection on `main`: "main protection gate" ruleset, both jobs required, empty bypass list. Direct pushes to `main` are rejected — everything goes through a branch + PR.

## Tunable constants

| Constant | File | Default | Governs |
|---|---|---|---|
| `SCORE_FLOOR` | `graph/helpers.py` | 0.7 | min hybrid-search score before `extract_info` |
| `SIMILARITY_THRESHOLD` | `db/semantic_cache.py` | 0.92 | cosine similarity for a cache hit |
| `MAX_RETRIES` | `graph/routing.py` | 2 | citation/search retries before fallback |
| clarification cap | `graph/routing.py` | 1 | per conversation, before forced search |
| `overlap_threshold` | `db/citation_verifier.py` | 0.3 | title overlap required to pass |
| `keep_turns` | `graph/helpers.py` | 6 | exchanges kept before pruning |
| `max_tokens` | `graph/nodes.py` | 4000 | history token budget for the agent |

## Testing

```bash
uv pip install -e .
cd src && pytest
```

8 test files: citation verification, retry/clarification routing edges, agent exception-recovery cascade, `stale_message_ids` pruning rules, retrieval metrics.

## Deployment

Render, single web service, `--workers 1` (non-negotiable — see Quickstart).

**Build:** `pip install -r requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"`

**Start:** `uvicorn api.api:app --host 0.0.0.0 --port $PORT --workers 1`

**Env:** `DATABASE_URL`, `CEREBRAS_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `FASTEMBED_CACHE_PATH` (avoids re-downloading the model on every cold start).

> Free-tier Render spins down after ~15 min idle — first request after that eats a cold start.

## License

MIT