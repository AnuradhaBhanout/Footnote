# Footnote
[![CI](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml)

**Most AI agents hand you whatever the model produced. This one checks the answer against what its tools actually returned — and refuses when they don't match.**

[Live demo](https://ragchatbot-ui-three.vercel.app) · [Frontend repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

Footnote is a LangGraph agent pipeline over a curated library of arXiv research papers. Retrieval is just one step; the rest of the graph enforces deterministic boundaries and verification checks on what the agent is allowed to do with what it retrieves:

- **Every citation is verified against real tool output before the user sees it.** An invented arXiv ID fails; a real ID paired with an invented title or finding also fails. Either failure routes the draft back with the specific discrepancy named in feedback, and after two retries the answer safely degrades to *"I don't have enough verified information"* instead of a plausible hallucination.
- **The pipeline invokes `extract_info` deterministically, exactly once.** Fetching paper details is not left to the agent's discretion; once search identifies relevant paper IDs (`score >= 0.7`), the pipeline executes extraction itself and runs a final synthesis prompt against the retrieved text.
- **Failure modes are bounded by code, not prompt instructions.** Recursion limits, retry caps, a single-clarification ceiling with a forced-search fallback, and a five-branch recovery cascade around the agent call ensure the model cannot loop or bypass safeguards.
- **Answers are cached only when the current turn earned it.** The reliability flag is recomputed per turn rather than inherited from session history — an answer derived solely from conversation memory without retrieval is never cached. Storage requires zero retries, and cache entries are fingerprinted against the current paper library so cached answers self-invalidate when the corpus changes.
- **Bounded conversation memory eliminates state explosion.** The entry node strips raw tool payloads and intermediate tool-calling messages from the checkpointed history, keeping only the user queries and final assistant answers up to a 6-turn sliding window.

Built on LangGraph for orchestration, MCP (Model Context Protocol) for tool serving, FastAPI/SSE for streaming, hybrid BM25 + dense embeddings over PostgreSQL with `pgvector`, and Langfuse for end-to-end tracing and online evaluation. Deployed as a single service on Render. The frontend (React + Vite) is hosted on Vercel; this repository contains the backend service. (Formerly named *RAGchatbot*).

---

## How It Works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> P["prune<br/><i>Strips tool payloads & trims checkpoint history to last 6 turns</i>"]
    P --> B["check_cache<br/><i>Semantic cosine lookup (&ge; 0.92) for current corpus version</i>"]

    B -- "Cache Hit" --> C["END<br/><i>Return verified cached answer</i>"]
    B -- "Cache Miss" --> D["run_agent<br/><i>LangChain agent with 5-branch recovery cascade</i><br/>• Tools: hybrid_search_papers, search_papers, ask_clarification<br/>• Disambiguation: grounded options / _force_search<br/>• Deterministic extraction: extract_info + synthesis"]

    D -- "Search Insufficient (retries < 2)" --> D
    D -- "Ambiguous Query (clarify_count = 0)" --> E["clarify<br/><i>LangGraph interrupt: pauses for user disambiguation</i>"]
    E -- "POST /resume" --> D

    D -- "Execution Complete" --> G["check_citations<br/><i>Validates arXiv IDs & checks title overlap (&ge; 0.3)</i>"]

    G -- "Passed" --> H["END<br/><i>Stream complete; cached in SSE layer if earned</i>"]
    G -- "Failed (retries < 2)" --> I["retry_with_feedback<br/><i>Injects exact citation mismatch into query</i>"]
    I --> D
    G -- "Failed (retries &ge; 2)" --> J["fallback<br/><i>'Not enough verified info' safety response</i>"]
    J --> H# Footnote
[![CI](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/AnuradhaBhanout/Footnote/actions/workflows/ci.yml)

**Most AI agents hand you whatever the model produced. This one checks the answer against what its tools actually returned — and refuses when they don't match.**

[Live demo](https://ragchatbot-ui-three.vercel.app) · [Frontend repo](https://github.com/AnuradhaBhanout/RAGchatbot-ui)

Footnote is a LangGraph agent pipeline over a curated library of arXiv research papers. Retrieval is just one step; the rest of the graph enforces deterministic boundaries and verification checks on what the agent is allowed to do with what it retrieves:

- **Every citation is verified against real tool output before the user sees it.** An invented arXiv ID fails; a real ID paired with an invented title or finding also fails. Either failure routes the draft back with the specific discrepancy named in feedback, and after two retries the answer safely degrades to *"I don't have enough verified information"* instead of a plausible hallucination.
- **The pipeline invokes `extract_info` deterministically, exactly once.** Fetching paper details is not left to the agent's discretion; once search identifies relevant paper IDs (`score >= 0.7`), the pipeline executes extraction itself and runs a final synthesis prompt against the retrieved text.
- **Failure modes are bounded by code, not prompt instructions.** Recursion limits, retry caps, a single-clarification ceiling with a forced-search fallback, and a five-branch recovery cascade around the agent call ensure the model cannot loop or bypass safeguards.
- **Answers are cached only when the current turn earned it.** The reliability flag is recomputed per turn rather than inherited from session history — an answer derived solely from conversation memory without retrieval is never cached. Storage requires zero retries, and cache entries are fingerprinted against the current paper library so cached answers self-invalidate when the corpus changes.
- **Bounded conversation memory eliminates state explosion.** The entry node strips raw tool payloads and intermediate tool-calling messages from the checkpointed history, keeping only the user queries and final assistant answers up to a 6-turn sliding window.

Built on LangGraph for orchestration, MCP (Model Context Protocol) for tool serving, FastAPI/SSE for streaming, hybrid BM25 + dense embeddings over PostgreSQL with `pgvector`, and Langfuse for end-to-end tracing and online evaluation. Deployed as a single service on Render. The frontend (React + Vite) is hosted on Vercel; this repository contains the backend service. (Formerly named *RAGchatbot*).

---

## How It Works

```mermaid
flowchart TD
    A["POST /chat (SSE)"] --> P["prune<br/><i>Strips tool payloads & trims checkpoint history to last 6 turns</i>"]
    P --> B["check_cache<br/><i>Semantic cosine lookup (&ge; 0.92) for current corpus version</i>"]

    B -- "Cache Hit" --> C["END<br/><i>Return verified cached answer</i>"]
    B -- "Cache Miss" --> D["run_agent<br/><i>LangChain agent with 5-branch recovery cascade</i><br/>• Tools: hybrid_search_papers, search_papers, ask_clarification<br/>• Disambiguation: grounded options / _force_search<br/>• Deterministic extraction: extract_info + synthesis"]

    D -- "Search Insufficient (retries < 2)" --> D
    D -- "Ambiguous Query (clarify_count = 0)" --> E["clarify<br/><i>LangGraph interrupt: pauses for user disambiguation</i>"]
    E -- "POST /resume" --> D

    D -- "Execution Complete" --> G["check_citations<br/><i>Validates arXiv IDs & checks title overlap (&ge; 0.3)</i>"]

    G -- "Passed" --> H["END<br/><i>Stream complete; cached in SSE layer if earned</i>"]
    G -- "Failed (retries < 2)" --> I["retry_with_feedback<br/><i>Injects exact citation mismatch into query</i>"]
    I --> D
    G -- "Failed (retries &ge; 2)" --> J["fallback<br/><i>'Not enough verified info' safety response</i>"]
    J --> H
```

### Execution Lifecycle

1. **Prune (`prune`)**: Runs at the entry point. Inspects thread history and removes all `ToolMessage` and tool-calling `AIMessage` objects, keeping only conversational exchanges up to the last 6 turns.
2. **Semantic Cache (`check_cache`)**: Computes the dense embedding of the query and checks the PostgreSQL `semantic_cache` table using cosine distance (`<=>`). Hits require cosine similarity $\ge 0.92$ and an exact match on `corpus_version` (an MD5 fingerprint of all sorted paper IDs in the library).
3. **Agent & Retrieval (`run_agent`)**:
   - `hybrid_search_papers`: Combines BM25 sparse keyword scores and `all-MiniLM-L6-v2` dense vector similarity ($\alpha = 0.5$).
   - **LLM Relevance Judge**: A separate Cerebras call (`gpt-oss-120b`) evaluates whether at least one candidate genuinely answers the query. For quoted title queries (`"..."`), it enforces a 0.7 title word-overlap check.
   - **Live arXiv Fallback**: If local search is insufficient or empty, the agent invokes `search_papers`, which searches the arXiv API, stores records in PostgreSQL, enqueues background embedding (`request_embed`), and passes candidates through the relevance judge.
   - **Grounded Clarification**: If the query is ambiguous, the agent calls `ask_clarification`. On the first request (`clarify_count = 0`), the pipeline generates grounded options from real candidate paper titles and yields a LangGraph `interrupt`. If the agent attempts a second clarification, `_force_search` deterministically executes search on the original query.
   - **Deterministic Extraction**: The agent does not call `extract_info`. Once search returns paper IDs meeting `SCORE_FLOOR = 0.7`, `run_agent` deterministically invokes `extract_info`, fetches the metadata, and runs a final summary pass instructing the model to summarize strictly from those details.
4. **Citation Verification (`check_citations`)**: Scans the draft answer for arXiv ID patterns (`\b\d{4}\.\d{4,5}(?:v\d+)?\b`). Verifies that each cited ID appeared in actual tool results and that at least 30% of significant title words appear in the answer text.
5. **Feedback & Fallback**:
   - If citation checks fail and `citation_retries < 2`, `retry_with_feedback` constructs a corrective prompt detailing the exact problem and re-invokes `run_agent`.
   - If citation retries reach 2, `fallback` returns a safe response (*"I don't have enough verified information to answer that accurately from your saved papers..."*).
6. **SSE Persistence**: In `api/sse.py`, if the response completed with zero search/citation retries, `answer_is_reliable = True`, and was not a cache hit, the verified answer and fetched papers are stored in `semantic_cache`.

---

## Features

- **Hybrid Retrieval with LLM-as-a-Judge**: Combines BM25 and dense embeddings (`fastembed` ONNX runtime) normalized with min-max scaling. An LLM evaluator judges candidate relevance before the agent proceeds.
- **Deterministic Tool Extraction**: The graph deterministically executes `extract_info` once search identifies qualifying candidate IDs, avoiding unnecessary tool-calling loops.
- **Dual-Phase Citation Verification**: Validates both the existence of cited arXiv IDs and title-word overlap ($\ge 0.3$) against real tool outputs. Distinguishes between verified passes and turns with nothing to verify.
- **Grounded, Bounded Clarification**: Ambiguous requests pause execution via LangGraph interrupts, offering options sourced from real paper titles. Clarification is strictly capped at one attempt per conversation.
- **Library-Fingerprinted Semantic Cache**: Verified answers are cached using query embeddings. Every entry is tagged with an MD5 hash of all indexed paper IDs, ensuring instant invalidation whenever papers are added or updated.
- **Sliding-Window Checkpoint Pruning**: `prune` strips ephemeral tool traffic (which accounts for ~95% of state size) and limits conversational history to 6 turns, preventing quadratic checkpoint growth in PostgreSQL.
- **Five-Branch Resilience Cascade**: Catches `GraphRecursionError`, MCP connection drops (`ClosedResourceError`, `McpError`), read timeouts (`httpx.ReadTimeout`), and malformed function-call parameters (`APIError`), recovering or providing bounded fallbacks.
- **In-Process MCP Architecture**: The FastMCP tool server is mounted directly into FastAPI at `/mcp` over SSE, eliminating external network boundaries between agent and tools.
- **Production Observability**: Live session tracing in Langfuse with deterministic scores (`cache_hit`, `error`, `citation_pass_rate`) and UI feedback integration (`user_feedback`).

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph, LangChain (`create_agent`) | Stateful graph execution, conditional routing, and interrupts |
| **Backend API** | FastAPI, Uvicorn | Async REST endpoints and Server-Sent Events (SSE) streaming |
| **Tool Protocol** | FastMCP / MCP | In-process tool server mounted at `/mcp` via SSE transport |
| **LLM Inference** | Cerebras API (`gpt-oss-120b`) | Agent reasoning, relevance evaluation, and final synthesis |
| **Vector & Sparse Search** | `fastembed` (`all-MiniLM-L6-v2`), `rank-bm25` | 384-dimensional dense vectors (ONNX) + BM25Okapi sparse retrieval |
| **Database & Cache** | PostgreSQL, `pgvector`, `psycopg3`, `psycopg2` | Paper metadata, IVFFlat embeddings index, HNSW semantic cache, and thread checkpoints |
| **Observability** | Langfuse | Distributed request tracing, session grouping, and automated scoring |
| **Rate Limiting** | `slowapi` | In-memory IP rate limiting (10 req/min) |
| **Frontend** | React, Vite (separate repository) | Interactive chat UI, citation inspector, and feedback buttons |

---

## Project Structure

```
Footnote/
├── pyproject.toml              # Project dependencies, build configuration, and dev extras
├── README.md                   # Technical documentation and architecture specification
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI: test, install, smoke test, and deploy hook
└── src/
    ├── api/
    │   ├── api.py              # FastAPI application, CORS, rate limits, /chat, /resume, /health
    │   ├── dependencies.py     # get_chatbot dependency with 503 readiness guard
    │   ├── schemas.py          # Pydantic models: ChatRequest, ResumeRequest, FeedbackRequest
    │   └── sse.py              # SSE event-loop generator and post-stream semantic cache storage
    ├── client/
    │   ├── agent_prompt.py     # System prompt rules, search tool filter (EXCLUDED_FROM_AGENT)
    │   ├── mcp_content.py      # Normalizes raw MCP content blocks to Python dictionaries
    │   ├── mcp_v1_chatBot.py   # MCP client lifecycle, connection retry, agent rebuild, pooling
    │   └── tools.py            # LangChain tool definition for ask_clarification
    ├── db/
    │   ├── citation_verifier.py# Extracts real paper titles from tool messages and verifies overlap
    │   ├── db.py               # psycopg2 connection pool, table schemas, and pgvector extension setup
    │   ├── embedding_model.py  # FastEmbed TextEmbedding wrapper with L2 normalization
    │   ├── paper_store.py      # Server-side streaming cursor for papers, corpus fingerprinting
    │   ├── rag_index.py        # HybridIndex: BM25 + dense vector index, upsert, embed_specific
    │   └── semantic_cache.py   # PostgreSQL-backed semantic cache with HNSW cosine search
    ├── evals/
    │   ├── queries.jsonl       # Frozen 20-query known-item evaluation set
    │   └── run_eval.py         # Offline eval runner: Recall@5, MRR, and alpha sweep
    ├── graph/
    │   ├── graph_pipeline.py   # StateGraph compilation and conditional routing topology
    │   ├── helpers.py          # Token counter, paper ID collector, stale_message_ids pruning
    │   ├── nodes.py            # GraphNodes: prune, check_cache, run_agent, clarify, check_citations
    │   ├── routing.py          # Routing predicates: after_cache, after_run_agent, after_citation_check
    │   └── state.py            # GraphState TypedDict with add_messages reducer
    ├── scripts/
    │   └── reembed.py          # One-off script to generate missing embeddings for existing papers
    ├── server/
    │   ├── index_state.py      # Process-global singletons: HybridIndex and SemanticCache
    │   ├── mcp_app.py          # FastMCP server instance and /health endpoint
    │   ├── relevance.py        # LLM relevance evaluator and quoted title extraction
    │   └── tools.py            # MCP tools: hybrid_search_papers, search_papers, extract_info, cache
    ├── tests/                  # Pytest test suite (48 unit and regression tests)
    ├── log_setup.py            # Logger configuration with rotating file handler
    ├── pytest.ini              # Pytest configuration (asyncio_mode = auto)
    ├── requirements.txt        # Frozen dependencies compiled by uv
    ├── run_dev.py              # Windows development launcher using asyncio SelectorEventLoop
    └── runtime.txt             # Python runtime specification (python-3.12.7)
```

---

## Getting Started

### Prerequisites

- **Python**: 3.12+
- **Database**: PostgreSQL with the `vector` (`pgvector`) extension enabled (e.g., Neon)
- **API Keys**:
  - Cerebras API key (required for LLM inference)
  - Langfuse API keys (optional, for tracing and scoring)

### 1. Clone and Install

Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```bash
git clone https://github.com/AnuradhaBhanout/Footnote.git
cd Footnote

# Install project with development dependencies
uv pip install -e ".[dev]"
```

### 2. Environment Configuration

Create a `.env` file in the root directory (or in `src/.env`):

```env
# PostgreSQL connection string with pgvector extension
DATABASE_URL=postgresql://user:password@ep-example.region.aws.neon.tech/dbname?sslmode=require

# Cerebras Inference API Key
CEREBRAS_API_KEY=csk-your-cerebras-key

# Langfuse Tracing (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional Server Settings
PORT=8000
ALLOWED_ORIGINS=http://localhost:5173
```

Tables (`papers`, `paper_embeddings`, `semantic_cache`, and LangGraph checkpoints) are created automatically on first startup via `init_db()` and `AsyncPostgresSaver.setup()`.

### 3. Run the Server

#### On Linux / macOS:
```bash
cd src
uvicorn api.api:app --host 0.0.0.0 --port 8000 --workers 1
```

#### On Windows:
On Windows, Uvicorn defaults to `ProactorEventLoop`, whereas `psycopg3`'s asynchronous pool requires a selector loop. Use the provided launcher:
```bash
cd src
python run_dev.py
```

> [!IMPORTANT]
> **Single Worker Required**: Run with `--workers 1`. `HybridIndex` and `SemanticCache` are process-global singletons held in memory; running multiple worker processes creates separate in-memory index copies without synchronized updates.

---

## API Reference

The backend exposes REST endpoints and Server-Sent Events (SSE) streaming. Rate limits are enforced at **10 requests per minute per IP** across all POST routes.

### Endpoints Overview

| Endpoint | Method | Response Type | Description |
|---|---|---|---|
| `/chat` | `POST` | `text/event-stream` | Primary streaming query interface |
| `/resume` | `POST` | `text/event-stream` | Resumes an interrupted clarification turn |
| `/feedback` | `POST` | `application/json` | Records user thumbs-up/down score in Langfuse |
| `/health` | `GET`, `HEAD` | `application/json` | Readiness and database connection check |
| `/mcp/sse` | `GET`, `POST` | `text/event-stream` | In-process MCP tool server endpoint |
| `/mcp/health` | `GET`, `HEAD` | `application/json` | MCP tool server health check |

---

### `POST /chat`

Streams execution events, tool calls, LLM tokens, and final answer payloads.

**Request:**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are recent techniques for citation faithfulness?",
    "session_id": null
  }'
```

**Payload Schema:**
- `query` (string, required): 1 to 2000 characters.
- `session_id` (UUID string, optional): Thread identifier for persistent conversation memory. If omitted, a UUID is generated.

---

### `POST /resume`

Resumes a session that paused on a clarification interrupt.

**Request:**
```bash
curl -N -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "answer": "CiteCheck: Towards Accurate Citation Faithfulness Detection"
  }'
```

**Payload Schema:**
- `session_id` (UUID string, required): Must correspond to a currently interrupted thread. Returns 404 if no paused execution exists.
- `answer` (string, required): 1 to 500 characters providing the user's clarification.

---

### SSE Stream Events

The stream delivers events formatted as `event: <name>\ndata: <json>\n\n`:

| Event | Data Fields | Trigger Condition |
|---|---|---|
| `tool_start` | `{"tool": string, "input": object}` | Agent initiates a tool call (`hybrid_search_papers`, `search_papers`, etc.) |
| `tool_end` | `{"tool": string, "input": object, "output": string}` | Tool call returns. Non-search outputs truncated to 300 chars. |
| `token` | `{"content": string}` | LLM streams an answer chunk |
| `interrupt` | `{"question": string, "options": string[], "session_id": string}` | Query is ambiguous; graph pauses for user selection |
| `done` | `{"answer": string, "session_id": string, "cited_paper_ids": string[], "fetched_papers": object[], "trace_id": string}` | Graph execution completed successfully |
| `error` | `{"message": string}` | Unhandled exception encountered during execution |

---

### `POST /feedback`

Submits human feedback associated with a specific request trace.

**Request:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "e9282201edb8bc3efc02b2a5ce6582fd",
    "is_positive": true
  }'
```

---

### `GET /health`

Returns readiness status and executes a live `SELECT 1` query to verify database connectivity.

**Response:**
```json
{
  "status": "ok",
  "ready": true,
  "db": true
}
```
*`ready` remains `false` during cold starts until the background MCP connection is established and the LangGraph graph is compiled.*

---

## Citation Verification Details

Implemented in `src/db/citation_verifier.py`, verification proceeds in two stages:

1. **Existence Verification**: Extracts all arXiv IDs cited in the answer using regex (`\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)\b`). Compares each normalized ID against the set of paper IDs returned by tools (`extract_info`, `hybrid_search_papers`, `search_papers`) during the turn.
2. **Title Word Overlap**: For each valid cited ID, compares significant title words (words $>3$ characters) against the answer text:
   $$\text{overlap} = \frac{|\text{title words in answer}|}{|\text{significant title words}|}$$
   If $\text{overlap} < 0.3$, the citation fails with an issue indicating the title or findings may be fabricated.

### The `verified` Flag vs `passed`
- An answer with no citations returns `passed = True, verified = False`.
- An answer where all cited papers match tool results returns `passed = True, verified = True`.
- `check_citations` logs `citation_pass_rate` to Langfuse **only when `verified = True`**. This prevents answers answered from memory without citations from artificially inflating the verification pass rate.

---

## State Pruning and Conversation Memory

LangGraph checkpoints preserve `messages` using the `add_messages` reducer. Without pruning, intermediate tool traffic and raw retrieval payloads accumulate across turns, degrading response latency and ballooning database checkpoint storage.

### Solution: Entry-Point Pruning

The `prune` node runs at the start of each turn using `stale_message_ids()`:
1. Strips all `ToolMessage` instances and tool-calling `AIMessage` objects.
2. Identifies human message indices and removes exchanges older than `keep_turns = 6`.
3. Emits LangGraph `RemoveMessage` commands to prune records from the thread checkpoint.

### Checkpoint Payload Reduction

Measured across consecutive versions of a live multi-turn thread:

| Checkpoint Version | `messages` Size | State Change |
|---|---|---|
| `...315` (Unpruned) | 409,798 bytes | Baseline with accumulated tool payloads |
| `...316` (Pruned) | 20,650 bytes | **95.0% reduction** upon wiring `prune` node |
| `...319` | 20,650 bytes | Stable size across subsequent turn |
| `...322` | 22,052 bytes | Maintained sliding window |

---

## Observability and Online Evaluation

Integrated with Langfuse via `CallbackHandler` in `api/sse.py`. Every request generates a trace tagged with `session_id`, `user_id`, and `tags=["chat"]` or `tags=["resume"]`.

### Deterministic Online Scores

| Score Name | Reported By | Range | Definition |
|---|---|---|---|
| `cache_hit` | `check_cache` node | `0` or `1` | `1` if answered via semantic cache; `0` on cache miss |
| `error` | `check_cache` / `run_agent` | `0` or `1` | `1` if execution failed or degraded to fallback |
| `citation_pass_rate` | `check_citations` node | `0` or `1` | `1` if all citations passed; only scored if `verified == True` |
| `user_feedback` | `POST /feedback` route | `0` or `1` | User thumbs-up (`1`) or thumbs-down (`0`) from UI |

---

## Tunable Constants

System parameters are maintained next to their respective logic:

| Constant | Location | Default | Description |
|---|---|---|---|
| `SCORE_FLOOR` | `graph/helpers.py` | `0.7` | Minimum hybrid search score for deterministic extraction |
| `SIMILARITY_THRESHOLD` | `db/semantic_cache.py` | `0.92` | Minimum cosine similarity for semantic cache hit |
| `MAX_RETRIES` | `graph/routing.py` | `2` | Maximum retry attempts for search insufficiency and citation checks |
| Clarification Cap | `graph/routing.py`, `graph/nodes.py` | `1` | Maximum clarification requests allowed per conversation |
| `overlap_threshold` | `graph/nodes.py`, `db/citation_verifier.py` | `0.3` | Minimum fraction of title words required in answer text |
| `keep_turns` | `graph/helpers.py` | `6` | Conversational exchange turns retained in checkpoint history |
| `max_tokens` (trim) | `graph/nodes.py` | `4000` | Token budget limit for messages passed to the agent |
| Rate Limit | `api/api.py` | `10/minute` | Per-IP request limit on POST routes |

---

## Testing

The test suite includes 48 unit and regression tests covering citation verification, agent recovery branches, routing edges, hybrid search filtering, pruning logic, and evaluation metrics.

```bash
# Run test suite with dev dependencies
uv run --extra dev --directory src pytest -q
```

### Test Suite Breakdown

- `test_citation_verifier.py`: Verifies positive citations and verifies that citation-less answers return `verified=False`.
- `test_routing.py`: Tests cache routing, retry caps (`MAX_RETRIES`), citation edge routing, and clarification limits.
- `test_invoke_agent_with_recovery.py`: Verifies the 5-branch exception recovery cascade (`GraphRecursionError`, `McpError`, timeout, malformed calls).
- `test_stale_message_ids.py`: Tests tool traffic stripping and sliding window turn truncation.
- `test_hybrid_search_papers.py`: Validates hybrid retrieval, evaluator verdict parsing, and quoted-title guard.
- `test_search_papers_relevance.py`: Validates live arXiv search filtering and relevance evaluator matching.
- `test_embed_specific.py`: Regression test ensuring embedding uses the paper summary column rather than the URL.
- `test_eval_metrics.py`: Validates Recall@K and Reciprocal Rank calculation.

---

## Retrieval Evaluation

An offline evaluation script runs against a frozen 20-query known-item dataset located at `src/evals/queries.jsonl`.

```bash
cd src
uv run python evals/run_eval.py
```

### Alpha Sweep Results

Evaluates the balance between BM25 sparse keyword search ($\alpha = 0.0$) and dense vector search ($\alpha = 1.0$):

| Metric | $\alpha = 0.0$ (BM25 only) | $\alpha = 0.25$ | $\alpha = 0.50$ (Default) | $\alpha = 0.75$ | $\alpha = 1.00$ (Dense only) |
|---|---|---|---|---|---|
| **Recall@5** | 0.850 | 0.950 | **1.000** | **1.000** | 0.950 |
| **MRR** | 0.677 | 0.756 | **0.883** | 0.852 | 0.842 |

Hybrid search ($\alpha = 0.5$) outperforms either isolated method (+0.206 MRR over BM25 alone, +0.041 MRR over dense alone). BM25 alone is clearly worst (-0.206 MRR); dense alone drops one paper out of the top 5 entirely.

Four queries rank the target paper below rank 1, in each case outranked by a topically adjacent paper in the corpus: two FinRL variants competing with each other, a citation-faithfulness query outranked by other RAG papers, and a clinical-NLP query outranked by an adjacent EHR paper. These are recorded as benchmark findings rather than tuned away.

**Evaluation Scope and Caveats:**
- **Known-item retrieval only**: Measures ranking accuracy when the target paper is guaranteed to be in the index. It does not measure broad topical queries ("what is new in RAG"), ambiguous queries meant to trigger `clarify`, or requests for unindexed papers.
- **Sample size ($n=20$)**: With 20 queries, a single query shift accounts for 0.05 Recall@5. Variations between $\alpha = 0.50$, $0.75$, and $1.00$ are within statistical noise; the primary statistically significant finding is the substantial margin over BM25-only.
- **Formulation bias**: The test set is frozen and expressed in authentic user phrasing. Queries formulated identically to abstract text measurably inflate BM25 performance (+0.04 MRR).

---

## Deployment

The backend deploys as a single web service on Render.

### Build and Start Commands

- **Build Command**:
  ```bash
  pip install -r src/requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"
  ```
- **Start Command**:
  ```bash
  cd src && uvicorn api.api:app --host 0.0.0.0 --port $PORT --workers 1
  ```

### Production Environment Variables

| Variable | Requirement | Purpose |
|---|---|---|
| `DATABASE_URL` | Required | PostgreSQL connection string with `pgvector` enabled |
| `CEREBRAS_API_KEY` | Required | Cerebras API key for `gpt-oss-120b` |
| `FASTEMBED_CACHE_PATH` | Recommended | FastEmbed defaults to `/tmp`, which does not survive container restarts. Setting this inside the project directory allows the build step to pre-bake the ONNX model (~180 MB) directly into the image |
| `LANGFUSE_PUBLIC_KEY` | Optional | Public key for Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | Optional | Secret key for Langfuse tracing |
| `LANGFUSE_HOST` | Optional | Langfuse host URL (defaults to cloud) |
| `ALLOWED_ORIGINS` | Optional | Comma-separated CORS allowed origins (defaults to `http://localhost:5173`) |
| `PORT` | Optional | Server port (injected by Render, defaults to `8000`) |

> [!NOTE]
> Deployed on Render's free tier, the service spins down after 15 minutes of inactivity. The first request incurs a cold start while the container spins up and loads the ONNX model into memory. Uptime monitors hitting `/health` can keep the instance and database connection pool warm.

---

## License

No license file is currently included in the repository. Adding an **MIT License** is recommended before public distribution.