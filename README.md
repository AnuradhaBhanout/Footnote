# Delve

A research assistant that searches a saved library of arXiv papers, answers questions about them, and checks its own citations before an answer is shown. Every paper ID and title in an answer is matched against real tool output before the answer ships; a mismatch triggers a retry, not a guess. Formerly named RAGchatbot.

The frontend is a separate repo: [RAGchatbot-ui](https://github.com/AnuradhaBhanout/RAGchatbot-ui) (React + Vite, deployed on Vercel). This repo is the backend.

## How it works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> B["check_cache<br/>semantic match against previously<br/>verified answers for the current paper library"]
    B -- hit --> C[return cached answer]
    B -- miss --> D["run_agent<br/>LangChain agent, tools: hybrid_search_papers,<br/>search_papers, ask_clarification"]
    D -- ambiguous query --> E["clarify<br/>interrupt, waits for user answer"]
    D -- paper_ids found --> F["extract_info<br/>one deterministic call, never left to the model"]
    E -- user answers --> D
    F --> G[check_citations]
    G -- passed --> H["finalize<br/>caches the verified answer"]
    G -- "failed, retries < 2" --> I["retry_with_feedback<br/>names the exact citation problem"]
    G -- "failed, retries = 2" --> J["fallback<br/>'not enough verified info'"]
    I --> D
```

`hybrid_search_papers` combines BM25 and dense embeddings (`fastembed`) over the saved library, then a separate LLM call judges whether any result actually answers the query, not just shares words with it. Only when that comes back empty does the agent fall back to a live arXiv search.

## Features

- **Hybrid retrieval with an LLM relevance judge**: BM25 + dense embeddings narrow the candidates, then a strict judge model decides if any of them is actually relevant before the agent is allowed to use them
- **Deterministic citation extraction**: the graph batches exactly one `extract_info` call itself once search settles on paper IDs, instead of leaving the model to decide how many times to fetch details
- **Post-hoc citation verification**: every paper ID and title an answer cites is checked against real tool output; a fabricated or mismatched citation triggers a corrective retry, then a safe fallback after two failures
- **Human-in-the-loop clarification**: an ambiguous query pauses the graph (a LangGraph interrupt) and asks the user to disambiguate before any search runs
- **Semantic caching**: only verified, non-fallback answers are cached, keyed to a fingerprint of the current paper library, so the cache invalidates itself the moment the library changes
- **Session-scoped agent context**: the LLM only sees the current turn's messages (`_current_turn_messages`), not the full cross-session history a Postgres checkpointer would otherwise replay into it
- **MCP session keepalive**: a background ping every 120s catches a stale connection to the tool server and reconnects before a real user request ever hits it

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite (separate repo), Vercel |
| Backend API | FastAPI, SSE streaming |
| Orchestration | LangGraph, LangChain (`create_agent`) |
| Tool protocol | MCP (Model Context Protocol) / FastMCP |
| Retrieval | BM25 (`rank-bm25`) + dense embeddings (`fastembed`, ONNX) |
| Storage | PostgreSQL + `pgvector` |
| Background jobs | Redis + `arq` (embeds new papers off the request path) |
| LLM | Cerebras `gpt-oss-120b`, used for both the agent and the relevance judge |
| Tracing | Langfuse |
| Deployment | Render, API and tool server as two separate services |

### Prerequisites

- Python 3.12+
- PostgreSQL with the `pgvector` extension
- Redis
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
REDIS_URL=redis://host:6379
CEREBRAS_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
```

Run both services (Postgres tables are created automatically on first start, via `db.init_db()`):

```bash
# terminal 1: the MCP tool server
python -m server.research_server

# terminal 2: the API
uvicorn api.api:app --host 0.0.0.0 --port 8000
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| API health | http://localhost:8000/health |
| Tool server (SSE) | http://localhost:8001/sse |
| Tool server health | http://localhost:8001/health |

Point the [frontend](https://github.com/AnuradhaBhanout/RAGchatbot-ui)'s `VITE_API_URL` at the API address above.

## Project structure

```
src/
├── api/
│   ├── api.py                  # FastAPI app: /chat, /resume, /health
│   ├── schemas.py               # ChatRequest, ResumeRequest
│   ├── dependencies.py          # get_chatbot(), 503s until the graph is ready
│   └── sse.py                   # SSE event loop, formats tool_start/tool_end/done/error
├── client/
│   ├── mcp_v1_chatBot.py        # MCP client, connection lifecycle, keepalive
│   ├── agent_prompt.py          # system prompt, EXCLUDED_FROM_AGENT tool filter
│   ├── tools.py                 # the ask_clarification tool
│   ├── mcp_content.py           # normalizes MCP's content-block shapes to plain dicts
│   └── cli.py                   # standalone terminal client, doesn't go through api.py
├── graph/
│   ├── graph_pipeline.py        # wiring only, builds the StateGraph
│   ├── nodes.py                 # check_cache, run_agent, clarify, check_citations, finalize
│   ├── routing.py                # conditional edges: retry / clarify / fallback logic
│   ├── state.py                  # GraphState TypedDict
│   └── helpers.py                # _current_turn_messages, paper-id collection, SCORE_FLOOR
├── server/
│   ├── research_server.py        # FastMCP entrypoint, mounts the SSE app + arq worker
│   ├── tools.py                  # hybrid_search_papers, search_papers, extract_info, cache tools
│   ├── relevance.py               # the LLM-as-judge relevance evaluator
│   ├── resources_and_prompts.py  # papers:// resources, /health route
│   ├── mcp_app.py                # shared FastMCP instance
│   ├── worker.py                  # arq worker: embeds newly fetched papers
│   └── index_state.py             # HybridIndex + SemanticCache singletons, Redis pool
├── db/
│   ├── db.py                      # connection pool, schema init
│   ├── rag_index.py               # HybridIndex: BM25 + pgvector search
│   ├── embedding_model.py         # fastembed wrapper
│   ├── paper_store.py             # load_all_papers, corpus fingerprint
│   ├── semantic_cache.py          # Postgres-backed answer cache, SIMILARITY_THRESHOLD
│   └── citation_verifier.py       # matches cited IDs/titles against real tool output
├── structured_outputs.py          # TriageAssessment, QueryReformulation, unused dead code
└── tests/                         # pytest suite
```

## API

Both endpoints stream Server-Sent Events; nothing here is a single JSON response.

### `POST /chat`

```json
{ "query": "Find work on citation faithfulness", "session_id": "optional-existing-session" }
```

### `POST /resume`

Answers a pending clarification for a session that's paused on one.

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

## Citation verification

`db/citation_verifier.py` pulls every real paper ID and title out of the turn's tool results (`extract_info`, `search_papers`, `hybrid_search_papers`), scans the draft answer for anything that looks like an arXiv ID, and checks two things: that the ID actually appeared in a tool result, and that a meaningful fraction of that paper's real title shows up somewhere in the answer text. The first check catches an invented paper ID outright; the second catches the harder case, a real ID attached to an invented title or finding. Either failure sends the draft back through `retry_with_feedback`, which names the specific problem in the next prompt rather than asking the model to simply "try again."

## Resilience

Both Render services run on the free tier, which spins down after roughly 15 minutes idle and can take 50+ seconds to wake. Three layers keep that from surfacing as a broken query:

- An external uptime monitor pings both services' `/health` route every few minutes, keeping the processes themselves awake.
- `MCP_ChatBot._keepalive` pings the live MCP session between the API and the tool server every 120 seconds, independent of `/health`, because that session can go stale on its own (an idle proxy connection, a tool-server redeploy) even while both processes report healthy. A failed ping sets the same reconnect path a real connection failure would use, before a user request ever reaches the dead connection.
- The frontend fires a fire-and-forget ping to both `/health` routes the moment the page loads, so the wake-up starts while the visitor is still reading the screen instead of after they hit send.

## Tunable constants

There's no single config file; these live next to the code they govern.

| Constant | File | Default | Governs |
|---|---|---|---|
| `SCORE_FLOOR` | `graph/helpers.py` | 0.7 | minimum hybrid-search score a result needs before it's eligible for `extract_info` |
| `SIMILARITY_THRESHOLD` | `db/semantic_cache.py` | 0.92 | cosine similarity a query needs to count as a semantic-cache hit |
| `MAX_RETRIES` | `graph/routing.py` | 2 | citation-check and search-insufficiency retries before falling back |
| keepalive interval | `client/mcp_v1_chatBot.py` | 120s | how often the MCP session gets pinged |
| `overlap_threshold` | `db/citation_verifier.py` | 0.3 (call site) | fraction of a cited paper's title that must appear in the answer text |

## Testing

```bash
uv pip install -e .
pytest src/tests -v
```

Covers the graph's retry/recovery path, hybrid search relevance, semantic cache hits, and the API's request handling.

## License

No license file yet. Add one, MIT is a reasonable default, before treating this as public.