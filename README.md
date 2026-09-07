# Footnote

**Most AI agents hand you whatever the model produced. This one checks the answer against what its tools actually returned — and refuses when they don't match.**

[Live demo](https://ragchatbot-ui-three.vercel.app) · [Frontend repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

Footnote is a LangGraph agent pipeline over a shared library of arXiv papers. Retrieval is one node in it; the rest of the graph exists to constrain what the agent is allowed to do with what it retrieves:

- **Every citation is verified against tool output before the user sees it.** An invented arXiv ID fails; a real ID wrapped around an invented title or finding also fails. Either one sends the draft back with the specific problem named, and after two failures the answer becomes *"not enough verified information"* instead of a plausible guess.
- **The graph calls `extract_info` itself, exactly once.** Fetching paper details is too important to leave to the model's discretion, so it's a deterministic step in the pipeline, not a tool the agent decides to call.
- **Failure modes are bounded by code, not by prompt instructions.** Recursion limits, retry caps, a one-clarification ceiling with a forced-search fallback, and a five-branch recovery cascade around the agent call — each one a control the model cannot talk its way past.
- **Answers are cached only when this turn earned it.** The reliability flag is set by the turn that produced the answer, not inherited from earlier in the session — a turn that answers from conversation history without retrieving is never cached. Zero retries required, and the key is fingerprinted against the current library so the cache self-invalidates when the corpus changes.

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
- **Human-in-the-loop clarification, bounded and grounded**: an ambiguous query pauses the graph (a LangGraph interrupt) and asks the user to disambiguate. The agent supplies only the question — the options it offers are built by the graph from real `hybrid_search_papers` results, so it cannot invent a paper title to put in front of the user. One clarification per conversation; if the agent tries to ask again, `_force_search` runs the search itself rather than stalling
- **Semantic caching, gated per turn**: an answer is cached only if `extract_info` actually ran on this turn's search results and the turn needed zero retries. `answer_is_reliable` is recomputed every turn rather than carried forward, so an answer written from conversation history alone never reaches the cache. Entries are keyed to a fingerprint of the current paper library and invalidate when it changes
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
├── evals/
│   ├── queries.jsonl           # frozen 20-query known-item eval set
│   └── run_eval.py             # recall@5 / MRR, alpha sweep
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

`--workers 1` is not optional. `HybridIndex` and `SemanticCache` are process-global singletons, so a second worker means a second copy of the index in memory and no shared in-process state.

**Environment**

`DATABASE_URL`, `CEREBRAS_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, plus:

| Variable | Why |
|---|---|
| `FASTEMBED_CACHE_PATH` | fastembed defaults to `/tmp`, which does not survive a restart, so the model would be re-downloaded on every cold start. Pointing it inside the project directory lets the build-step prefetch above bake the model (~180 MB) into the image instead |

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

44 tests across six files: `hybrid_search_papers` behaviour including the quoted-title guard, `search_papers` relevance filtering, the agent's five-branch exception-recovery cascade in `_invoke_agent_with_recovery`, every conditional edge in `graph/routing.py` (both sides of the retry cap and the clarification cap), a regression test pinning `embed_specific` to the summary column, and the retrieval metric functions.

## Retrieval evaluation

A frozen query set lives at `src/evals/queries.jsonl`: 20 natural-language queries, each labelled with the one paper in the corpus that answers it. `src/evals/run_eval.py` runs them through `HybridIndex.search` and reports recall@5 and MRR, plus a per-query breakdown showing where the correct paper ranked and what outranked it.

```bash
cd src && uv run python evals/run_eval.py
```

Alpha sweep over the same loaded index (`alpha=0.0` is BM25 only, `1.0` is dense only):

| alpha | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| recall@5 | 0.850 | 0.950 | **1.000** | **1.000** | 0.950 |
| MRR | 0.677 | 0.756 | **0.883** | 0.852 | 0.842 |

Hybrid beats either component alone, which is the justification for `alpha=0.5` in `hybrid_search_papers`. BM25 alone is clearly worst (-0.21 MRR); dense alone drops one paper out of the top 5 entirely.

Four queries rank the correct paper below position 1, all beaten by topically adjacent papers in the same corpus: two FinRL variants competing with each other, a citation-faithfulness query outranked by other RAG papers, and a clinical-NLP query outranked by an adjacent EHR paper. These are recorded rather than tuned away.

**What this measures, and what it does not.** Known-item retrieval only: one correct paper per query, and that paper is known to be in the index. It cannot score broad topical queries ("what's new in RAG"), bare acronyms that should trigger the `clarify` node, or requests for papers absent from the corpus. Those need a differently-labelled set.

Caveats worth stating plainly. n=20 is small enough that one query moving is 0.05 recall, so the ordering among alpha 0.5/0.75/1.0 is within noise — the reliable finding is the gap to BM25-only. Some queries were drafted with model assistance from the same abstracts the index is built on, which biases toward easier retrieval. Queries phrased close to abstract wording measurably inflate the BM25-only column: reverting one such query dropped that column by 0.04 MRR with no code change. The set is therefore written in user phrasing and frozen — it changes when a label is wrong, never because retrieval failed.

## Known limitations

Stated rather than hidden, because they shape what the system can and can't do:

- **Retrieval eval is known-item only.** The 20-query set at `src/evals/` measures whether a known paper ranks in the top 5. There is no set-valued labelling for topical queries, no corpus-coverage measure, and no eval for the clarification path. Production quality is still observed through Langfuse scores (`cache_hit`, `citation_pass_rate`).
- **No rate limiting.** Under concurrent load, a 429 from the LLM provider degrades to a fallback answer rather than being queued or retried with backoff.
- **CORS is narrowed, but it protects the user, not the API.** `allow_origins` now reads from `ALLOWED_ORIGINS` (`api/api.py`), defaulting to localhost. Worth being precise about what that buys: CORS is enforced by the browser, so it does nothing against scripted abuse, and with no sessions or auth there is no credential to hijack. The real exposure is quota theft — someone pointing their own frontend at this backend — and the control for that is rate limiting, not CORS. The allowlist is set because it costs one line and is a precondition for adding auth later.

## License

No license file yet. Add one, MIT is a reasonable default, before treating this as public.