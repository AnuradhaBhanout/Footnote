# RAG Chatbot with MCP & LLM-as-a-Judge

This is a project I built to experiment with the Model Context Protocol (MCP) and LangChain. It is a CLI-based chatbot that does hybrid-search RAG over academic papers. 

To keep the chatbot's answers accurate, I implemented an "LLM-as-a-Judge" loop. When you search for something, a secondary model evaluates whether the retrieved local papers actually answer your question. If they don't, the chatbot automatically triggers a search on arXiv, downloads relevant papers, indexes them on the fly, and uses them to answer your query.

---

## How It Works

- **Decoupled Servers (MCP):** The chatbot client connects to three separate MCP servers: a custom **Research Server** (for searching and indexing papers), the community **Fetch Server** (for retrieving web pages), and a sandboxed **Filesystem Server** (for reading and writing files).
- **Hybrid Search:** Local searches combine semantic search (using `all-MiniLM-L6-v2` embeddings) with keyword search (using `BM25Okapi`). This gives much more accurate matches than using vector search alone.
- **Relevance Judge:** When you search, a cheap/fast LLaMA model evaluates the top results. If it finds they are irrelevant, it flags them as insufficient.
- **Self-Correction:** If the judge flags local results, the agent automatically runs an online arXiv search, downloads the papers' metadata, updates the local cache, and answers your question.
- **Caching:** The index is cached on disk as `_rag_index.pk1`. It only rebuilds when it detects that new papers have been added, making searches incredibly fast (under 300ms).

---

## File Structure

Here is how the project files are organized:

```text
RAGchatbot/
├── requirements.txt         # Project dependencies
├── server_config.json       # Config showing MCP server commands & paths
├── .env.example             # Template for your API keys
├── papers/                  # Where downloaded papers and the cache live
└── src/
    ├── mcp_v1_chatBot.py    # The main chatbot CLI loop
    ├── research_server.py   # FastMCP server that exposes tools to the LLM
    └── rag_index.py         # The hybrid search & caching engine