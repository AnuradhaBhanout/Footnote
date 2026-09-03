# Delve

**Most AI agents hand you whatever the model produced. This one checks the answer against what its tools actually returned — and refuses when they don't match.**

[Live demo](https://ragchatbot-ui-three.vercel.app) · [Frontend repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

Delve is a LangGraph agent pipeline over a shared library of arXiv papers. Retrieval is one node in it; the rest of the graph exists to constrain what the agent is allowed to do with what it retrieves:

- **Every citation is verified against tool output before the user sees it.** An invented arXiv ID fails; a real ID wrapped around an invented title or finding also fails. Either one sends the draft back with the specific problem named, and after two failures the answer becomes *"not enough verified information"* instead of a plausible guess.
- **The graph calls `extract_info` itself, exactly once.** Fetching paper details is too important to leave to the model's discretion, so it's a deterministic step in the pipeline, not a tool the agent decides to call.
- **Failure modes are bounded by code, not by prompt instructions.** Recursion limits, retry caps, a one-clarification ceiling with a forced-search fallback, and a five-branch recovery cascade around the agent call — each one a control the model cannot talk its way past.
- **Answers are cached only when the pipeline says they're trustworthy.** Reliable flag set, zero retries, keyed to a fingerprint of the current library so it self-invalidates the moment the corpus changes.

Built on LangGraph for orchestration, MCP for the tool layer, FastAPI/SSE for streaming, with hybrid BM25 + dense retrieval over pgvector and Langfuse tracing end to end. Deployed as a single Render service. The frontend (React + Vite, on Vercel) lives in a separate repo; this one is the backend. Formerly named RAGchatbot.

## How it works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> B["check_cache<br/>semantic match against previously<br/>verified answers for the current paper library"]
    B -- hit --> C[return cached answer]
    B -- miss --> D["run_agent<br/>LangChain agent, tools: hybrid_search_papers,<br/>search_papers, ask_clarification"]
    D -- "ambiguous, first time" --> E["clarify<br/>interrupt, waits for user answer"]
    D -- "ambiguous again" --> K["_force_search<br/>deterministic search on the original query"]
    D -- paper_ids found --> F["extract_info<br/>one deterministic call, never left to the model"]
    E -- user answers --> D
    K --> F
    F --> G[check_citations]
    G -- passed --> H["END<br/>verified answer cached in the SSE layer"]
    G -- "failed, retries < 2" --> I["retry_with_feedback<br/>names the exact citation problem"]
    G -- "failed, retries = 2" --> J["fallback<br/>'not enough verified info'"]
    I --> D
```

`hybrid_search_papers` combines BM25 and dense embeddings (`fastembed`) over the indexed library, then a separate LLM call judges whether any result actually answers the query, not just shares words with it. Only when that comes back empty does the agent fall back to a live arXiv search.

The graph has six nodes: `check_cache`, `run_agent`, `clarify`, `check_citations`, `retry_with_feedback`, `fallback`. Caching a verified answer happens in the SSE layer after the graph finishes, not in a graph node.

## Features

- **Hybrid retrieval with an LLM relevance judge**: BM25 + dense embeddings narrow the candidates, then a strict judge model decides if any of them is actually relevant before the agent is allowed to use them
- **Deterministic citation extraction**: the graph batches exactly one `extract_info` call itself once search settles on paper IDs, instead of leaving the model to decide how many times to fetch details
- **Post-hoc citation verification**: every paper ID and title an answer cites is checked against real tool output; a fabricated or mismatched citation triggers a corrective retry, then a safe fallback after two failures
- **Human-in-the-loop clarification, bounded**: an ambiguous query pauses the graph (a LangGraph interrupt) and asks the user to disambiguate. One clarification per conversation — if the agent tries to ask a second time, `_force_search` calls `hybrid_search_papers` directly on the original query instead of stalling
- **Semantic caching**: an answer is cached only when it is marked reliable and needed zero retries, keyed to a fingerprint of the current paper library, so the cache invalidates itself the moment the library changes
- **Session-scoped agent context**: the LLM only sees the current turn's messages (`_current_turn_messages`), not the full cross-session history a Postgres checkpointer would otherwise replay into it
- **In-process MCP**: the FastMCP tool server is mounted inside the FastAPI app at `/mcp`, so the agent's tool calls stay on loopback instead of crossing a network boundary between two services

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite (separate repo), Vercel |
| Backend API | FastAPI, SSE streaming |
| Orchestration | LangGraph, LangChain (`create_agent`) |
| Tool protocol | MCP (Model Context Protocol) / FastMCP, mounted in-process |
| Retrieval | BM25 (`rank-bm25`) + dense embeddings (`fastembed`, ONNX) |
| Storage | PostgreSQL + `pgvector` (Neon) |
| LLM | Cerebras `gpt-oss-120b`, used for both the agent and the relevance judge |
| Tracing | Langfuse |
| Deployment | Render, single web service |

### Prerequisites

- Python 3.12+
- PostgreSQL with the `pgvector` extension
- API keys: Cerebras, Langfuse (optional but wired in throughout)

### Run locally

```bash
git clone https://github.com/AnuradhaBhanout/RAGchatbot.git
cd RAGchatbot
uv pip install -e .
```

`.env`:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
CEREBRAS_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
```

One process serves both the API and the MCP tool server. Postgres tables are created automatically on first start via `db.init_db()`.

```bash
cd src
uvicorn api.api:app --host 0.0.0.0 --port 8000 --workers 1
```

On Windows, use `python run_dev.py` instead. Uvicorn hardcodes `ProactorEventLoop` there, and psycopg3's async pool requires a selector loop; `run_dev.py` starts the server under the right loop factory.

Run a single worker. `HybridIndex` and `SemanticCache` are process-global singletons, so multiple workers means multiple copies of the index and no shared in-process state.

| Endpoint | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| API health | http://localhost:8000/health |
| MCP SSE | http://localhost:8000/mcp/sse |
| MCP health | http://localhost:8000/mcp/health |

Point the [frontend](https://github.com/AnuradhaBhanout/RAGchatbot-ui)'s `VITE_API_URL` at the API address above.

`MCP_URL` is an optional override. Unset, the client connects to its own process on loopback. Set it only if you split the tool server back out into a separate service.

## Project structure

```
src/
├── api/
│   ├── api.py                  # FastAPI app: /chat, /resume, /health; mounts MCP at /mcp
│   ├── schemas.py              # ChatRequest, ResumeRequest
│   ├── dependencies.py         # get_chatbot(), 503s until the graph is ready
│   └── sse.py                  # SSE event loop; also stores verified answers in the cache
├── client/
│   ├── mcp_v1_chatBot.py       # MCP client, connection lifecycle, agent rebuild on reconnect
│   ├── agent_prompt.py         # system prompt, EXCLUDED_FROM_AGENT tool filter
│   ├── tools.py                # the ask_clarification tool
│   └── mcp_content.py          # normalizes MCP's content-block shapes to plain dicts
├── graph/
│   ├── graph_pipeline.py       # wiring only, builds the StateGraph
│   ├── nodes.py                # check_cache, run_agent, clarify, check_citations,
│   │                           #   retry_with_feedback, fallback, _force_search
│   ├── routing.py              # conditional edges: retry / clarify / fallback logic
│   ├── state.py                # GraphState TypedDict
│   └── helpers.py              # _current_turn_messages, paper-id collection, SCORE_FLOOR
├── server/
│   ├── mcp_app.py              # shared FastMCP instance, /health route
│   ├── tools.py                # hybrid_search_papers, search_papers, extract_info, cache tools
│   ├── relevance.py            # the LLM-as-judge relevance evaluator
│   └── index_state.py          # HybridIndex + SemanticCache singletons, background embed
├── db/
│   ├── db.py                   # connection pool, schema init
│   ├── rag_index.py            # HybridIndex: BM25 + dense search
│   ├── embedding_model.py      # fastembed wrapper, model loaded lazily on first use
│   ├── paper_store.py          # load_all_papers, corpus fingerprint
│   ├── semantic_cache.py       # Postgres-backed answer cache, SIMILARITY_THRESHOLD
│   └── citation_verifier.py    # matches cited IDs/titles against real tool output
├── run_dev.py                  # Windows-only local launcher (selector event loop)
└── tests/                      # pytest suite
```

## API

Both endpoints stream Server-Sent Events; nothing here is a single JSON response.

### `POST /chat`

```json
{ "query": "Find work on citation faithfulness", "session_id": "optional-existing-session" }
```

### `POST /resume`

Answers a pending clarification for a session that's paused on one. Returns 404 if no paused session exists for that ID.

```json
{ "session_id": "the-session-that-paused", "answer": "Temperature as a sampling hyperparameter" }
```

### Event stream

| Event | Payload | Sent when |
|---|---|---|
| `tool_start` | `{tool, input}` | the agent calls a tool |
| `tool_end` | `{tool, output, input}` | a tool call returns |
| `interrupt` | `{question, options, session_id}` | the graph pauses for clarification |
| `token` | `{content}` | the model streams a response token |
| `done` | `{answer, session_id, cited_paper_ids, fetched_papers, trace_id}` | the graph finishes |
| `error` | `{message}` | anything in the stream raised |

### `GET /health`

```json
{ "status": "ok", "ready": true, "db": true }
```

`ready` is false until the MCP session is connected and the graph is compiled. `db` runs a real `SELECT 1` rather than returning a constant, so an uptime monitor hitting this route also keeps a scale-to-zero Postgres warm.

## Citation verification

`db/citation_verifier.py` pulls every real paper ID and title out of the turn's tool results (`extract_info`, `search_papers`, `hybrid_search_papers`), scans the draft answer for anything that looks like an arXiv ID, and checks two things: that the ID actually appeared in a tool result, and that a meaningful fraction of that paper's real title shows up somewhere in the answer text. The first check catches an invented paper ID outright; the second catches the harder case, a real ID attached to an invented title or finding. Either failure sends the draft back through `retry_with_feedback`, which names the specific problem in the next prompt rather than asking the model to simply "try again."

An answer that cites nothing passes this check by construction — there is nothing to verify. Verification tells you the cited papers are real and were returned by a tool. It does not tell you they answer the question.

## Deployment

One Render web service.

**Build command**

```
pip install -r requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"
```

**Start command**

```
uvicorn api.api:app --host 0.0.0.0 --port $PORT --workers 1
```

**Environment**

`DATABASE_URL`, `CEREBRAS_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, plus:

| Variable | Why |
|---|---|
| `FASTEMBED_CACHE_PATH` | fastembed defaults to `/tmp`, which does not survive a restart, so the model would be re-downloaded on every cold start. Point it inside the project directory and the build-step prefetch above bakes it into the image |
| `MALLOC_ARENA_MAX=2` | the code is thread-heavy via `asyncio.to_thread`; capping glibc's per-thread arenas meaningfully lowers RSS |

Deployed on Render's free tier, which spins down after roughly 15 minutes idle. The first request after that incurs a cold start while the container boots and the ONNX model loads from disk. Production would run on always-on compute.

## Tunable constants

There's no single config file; these live next to the code they govern.

| Constant | File | Default | Governs |
|---|---|---|---|
| `SCORE_FLOOR` | `graph/helpers.py` | 0.7 | minimum hybrid-search score a result needs before it's eligible for `extract_info` |
| `SIMILARITY_THRESHOLD` | `db/semantic_cache.py` | 0.92 | cosine similarity a query needs to count as a semantic-cache hit |
| `MAX_RETRIES` | `graph/routing.py` | 2 | citation-check and search-insufficiency retries before falling back |
| clarification cap | `graph/routing.py`, `graph/nodes.py` | 1 | clarifications allowed per conversation before the graph forces a search |
| `overlap_threshold` | `db/citation_verifier.py` | 0.3 (call site) | fraction of a cited paper's title that must appear in the answer text |

## Testing

```bash
uv pip install -e .
cd src && pytest
```

24 tests across three files: `hybrid_search_papers` behaviour including the quoted-title guard, `search_papers` relevance filtering, and the agent's five-branch exception-recovery cascade in `_invoke_agent_with_recovery`.

## Known limitations

Stated rather than hidden, because they shape what the system can and can't do:

- **No offline retrieval eval.** Quality is observed through Langfuse scores (`cache_hit`, `citation_pass_rate`) in production. There is no frozen query set with expected paper IDs, so a retrieval change cannot be regressed against a number.
- **Clarification options aren't verified.** `ask_clarification` arguments are model-generated and don't pass through `citation_verifier`, so the options it offers can name papers that aren't in the library.
- **No rate limiting.** Under concurrent load, a 429 from the LLM provider degrades to a fallback answer rather than being queued or retried with backoff.
- **CORS is open.** `allow_origins=["*"]` in `api/api.py` should be narrowed to the frontend's domain.

## License

No license file yet. Add one, MIT is a reasonable default, before treating this as public.