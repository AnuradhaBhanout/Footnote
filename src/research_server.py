# %%writefile mcp_project/research_server.py
import sys
import logging
import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from semantic_cache import SemanticCache
import asyncio

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from rag_index import HybridIndex,load_all_papers
from openai import OpenAI

from db import init_db,get_conn

init_db()               # creates tables on first run, safe to call every time

# logging.basicConfig(
#     filename="research_server_debug.log",
#     level=logging.INFO,
#     format="%(asctime)s %(message)s",
# )
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/research_server_debug.log"),
        logging.StreamHandler() # This also prints to your terminal
    ]
)

_hybrid_index = HybridIndex()
_index_loaded = False

_semantic_cache = SemanticCache(model=_hybrid_index.model) # reuse the loaded model

# _evaluator_client =OpenAI(
#     base_url = "https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

_evaluator_client =OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

EVALUATOR_MODEL = "llama-3.1-8b-instant"

def evaluate_relevance(query:str,results:list)->dict:
    """LLM-as-judge: is at least one retrieved paper actually relevant, or is this a bad batch?"""
    if not results:
        return{"sufficient": False,"best_paper_id": None,"reason":"No results retrieved."}
    
    candidates = "\n".join(f"- {r['paper_id']}: {r['title']}" for r in results)

    prompt = f"""You are a strict relevance judge. Query: "{query}"

    Retrieved papers:{candidates}
    Does AT LEAST ONE paper genuinely answer the query - not just share the few words with it?
    Respond with ONLY this JSON, nothing else:
    {{"sufficient":true or false,"best_paper_id":"<id or null>","reason":"<one sentence>"}}
    """
    response = _evaluator_client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=150,
    )

    try:
        # return json.loads(response.choices[0].message.content)
        #response = _relevance_judge.invoke(prompt)
        content = response.choices[0].message.content.strip()
        
        # --- STRIP MARKDOWN BACKTICKS FOR ROBUST PARSING ---
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        # ---------------------------------------------------
        return json.loads(content)
    
    except (json.JSONDecodeError,AttributeError) as e:
        # Log the raw text to see what the LLM returned on parse failure
        logging.error(f"[evaluator parse error]: {str(e)} - raw text: {response.choices[0].message.content if response.choices else 'No choices available'}")
        return{"sufficient":False,
               "best_paper_id":results[0]["paper_id"]if results else None,
               "reason":"Judge Parse failure - defaulted to top result."}





#Its primary purpose is to defer expensive operations (such as loading deep learning models, reading files, or generating vector embeddings) 
# until the exact moment a user performs their first search, rather than doing it when the server boots up.
def _ensure_index_loaded():
    global _index_loaded
    if not _index_loaded:
        _hybrid_index.load_cache_or_build()
        _index_loaded = True


PAPER_DIR = "papers"

# Initialize FastMCP server
# mcp = FastMCP("research", host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
port = int(os.environ.get("PORT") or 8001)
mcp = FastMCP("research", host="0.0.0.0", port=port)
# mcp = FastMCP("research")

#Call LLM for dynamic search across all saved papers on disk using hybrid search architecture.
@mcp.tool()
async def hybrid_search_papers(query: str,top_k: int = 5,alpha: float = 0.5)->List[dict]:
    
    """
    Search ALL previously saved papers using hybrid retrieval (BM25 + embeddings),
    then have a separate judge model verify the results are actually relevant.

    Returns:
       Returns: {"results": [...], "evaluator_verdict": {"sufficient": bool, "best_paper_id": str, "reason": str}}
    """


    #If this is the first search of the session, it loads the embedding model and the cached papers into memory. 
    # If it has already run, it skips the step to maintain fast response times.
    #_ensure_index_loaded()
    await asyncio.to_thread(_ensure_index_loaded)

    # Checks if new papers have been downloaded to your disk directory (via the standard search_papers tool) since the search index was last built. 
    # If it detects changes, it automatically regenerates the vector embeddings and updates the BM25 dictionary on-the-fly.
    #_hybrid_index.refresh_if_stale()
    await asyncio.to_thread(_hybrid_index.refresh_if_stale)

    # Executes the underlying hybrid search 
    results = _hybrid_index.search(query,top_k=top_k,alpha=alpha)

    papers = load_all_papers()

    #print(f"\n[hybrid_search debug] query='{query}' alpha={alpha}",file=sys.stderr)
    logging.info(f"[hybrid_search debug] query='{query}' alpha={alpha}")
    for r in results:
        r["title"] = papers.get(r["paper_id"], {}).get("title","Unknown") 
        # print(f"  {r['paper_id']} | combined={r['score']:.3f} | dense={r['dense_score']:.3f} | bm25={r['bm25_score']:.3f} | {r['title']}",file=sys.stderr)
        logging.info(f"  {r['paper_id']} | combined={r['score']:.3f} | dense={r['dense_score']:.3f} | bm25={r['bm25_score']:.3f} | {r['title']}")

    judgment = evaluate_relevance(query,results)
    logging.info(f"[evaluator] sufficient={judgment.get('sufficient')} reason={judgment.get('reason')}")

    return {"results": results,"evaluator_verdict": judgment}                                                                               




@mcp.tool()
async def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    max_results = min(max_results, 10)
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    #papers = client.results(search)
    try:
      papers = await asyncio.wait_for( asyncio.to_thread(lambda: list(client.results(search))), timeout=45.0)
    except asyncio.TimeoutError:
        logging.error(f"search_papers: arxiv timed out for topic '{topic}'")
        return
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    # try:
    #     with open(file_path, "r") as json_file:
    #         papers_info = json.load(json_file)
    # except (FileNotFoundError, json.JSONDecodeError):
    #     papers_info = {}

    papers_info = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Assuming you write { "paper_id": { ... } } or similar structure
                    papers_info.update(record)
    except FileNotFoundError:
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    try:
        # with open(file_path, "w") as json_file:
        #     json.dump(papers_info, json_file, indent=2)
        # print(f"Results are saved in: {file_path}", file=sys.stderr)
        with open(file_path, "w") as json_file:
           for pid, info in papers_info.items():
               json_file.write(json.dumps({pid: info}) + "\n")
    except OSError:
        print("Disk write skipped (read-only filesystem)", file=sys.stderr)

    await asyncio.to_thread(_insert_papers_sync, papers_info, topic)
    return paper_ids



def _insert_papers_sync(papers_info: dict, topic: str):
    conn = get_conn()
    with conn:
        with conn.cursor() as cur:
            for pid, info in papers_info.items():
                cur.execute("""
                    INSERT INTO papers (paper_id, topic, title, authors, summary, pdf_url, published)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (paper_id) DO NOTHING
                """, (pid, topic, info['title'], json.dumps(info['authors']),
                    info['summary'], info['pdf_url'], info['published']))
    conn.close()
    
     #  rebuilt the index when new papers are saved(commented because it needs more space which does not cover under free render tier)
    #_ensure_index_loaded()
    #_hybrid_index.build()
    
    #return paper_ids

# @mcp.tool()
# def extract_info(paper_id: str) -> str:
#     """
#     Search for information about a specific paper across all topic directories.
    
#     Args:
#         paper_id: The ID of the paper to look for
        
#     Returns:
#         JSON string with paper information if found, error message if not found
#     """
 
#     for item in os.listdir(PAPER_DIR):
#         item_path = os.path.join(PAPER_DIR, item)
#         if os.path.isdir(item_path):
#             file_path = os.path.join(item_path, "papers_info.json")
#             if os.path.isfile(file_path):
#                 try:
#                     with open(file_path, "r") as json_file:
#                         papers_info = json.load(json_file)
#                         if paper_id in papers_info:
#                             return json.dumps(papers_info[paper_id], indent=2)
#                 except (FileNotFoundError, json.JSONDecodeError) as e:
#                     print(f"Error reading {file_path}: {str(e)}",file=sys.stderr)
#                     continue
    
#     return f"There's no saved information related to paper {paper_id}."


@mcp.tool()
async def extract_info(paper_id: str) -> str:

    def _fetch():
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, authors, summary, pdf_url, published FROM papers WHERE paper_id = %s",
                (paper_id,)
            )
            row = cur.fetchone()
        conn.close()
        return row
   
    row = await asyncio.to_thread(_fetch)
    if not row:
        return f"There's no saved information related to paper {paper_id}."

    return json.dumps({
        "title": row[0],
        "authors": row[1],
        "summary": row[2],
        "pdf_url": row[3],
        "published": row[4],
    }, indent=2)

# Adding resources 
# @mcp.resource("papers://folders")
# def get_available_folders() -> str:
#     """
#     List all available topic  folders  in the papers directory.

#     This resource provide a simple list of all available topic folders.
#     """
#     folders = []

#     # Get all topic directories
#     if os.path.exists(PAPER_DIR):
#         for topic_dir in os.listdir(PAPER_DIR):
#             topic_path = os.path.join(PAPER_DIR,topic_dir)
#             if os.path.isdir(topic_path):
#                 papers_file = os.path.join(topic_path,"papers_info.json")
#                 if os.path.exists(papers_file):
#                     folders.append(topic_dir)


#     # Create a simple markdown list
#     content = "# Available Topics\n\n"
#     if folders:
#         for folder in folders:
#             content += f"- {folder}\n"
#         content += f"\nUse @{folder} to access papers in that topic.\n"
    
#     else:
#         content += "No topics found.\n"

#     return content

@mcp.resource("papers://folders")
async def get_available_folders() -> str:
    def _fetchfolder():  
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT topic FROM papers ORDER BY topic;")
            topics = [r[0] for r in cur.fetchall()]
        conn.close()
        return topics
    
    topics = await asyncio.to_thread(_fetchfolder)

    content = "# Available Topics\n\n"
    if topics:
        for topic in topics:
            content += f"- {topic}\n"
        content += f"\nUse @{topics[-1]} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"
    return content


# @mcp.resource("papers://{topic}")
# def get_topic_papers(topic: str) -> str:
#     """
#     Get detailed  information about papers on a specific topic.

#     Args:
#         topic : The research topic to retrieve papers for
#     """
#     topic_dir = topic.lower().replace(" ","_")
#     papers_file = os.path.join(PAPER_DIR,topic_dir,"papers_info.json")

#     if not os.path.exists(papers_file):
#        return f" # No papers found for topic: {topic}\n\nTry searching for papers on this topic first"
    
#     try:
#         with open(papers_file,'r')as f:
#             papers_data =json.load(f)

#         #Create content with paper details
#         content = f"# Papers on {topic.replace('_',' ').title()}\n\n"
#         content += f"Total Papers: {len(papers_data)}\n\n"

#         for paper_id, paper_info in papers_data.items():
#             content += f"## {paper_info['title']}\n"
#             content += f"- **Paper ID**: {paper_id}\n"
#             content += f"- **Authors**: {'. '.join(paper_info['authors'])}\n"
#             content += f"- **Published**: {paper_info['published']}\n"
#             content += f"- **PDF URL**: [{paper_info['pdf_url']}]   ({paper_info['pdf_url']})\n\n"
#             content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
#             content += "---\n\n" 
            
#         return content
#     except json.JSONDecodeError:
#         return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."


@mcp.resource("papers://{topic}")
async def get_topic_papers(topic: str) -> str:
    def _fetchpapers():
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT paper_id, title, authors, summary, pdf_url, published FROM papers WHERE topic = %s",
                (topic.lower().replace(" ", "_"),)
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    
    rows = await asyncio.to_thread(_fetchpapers)
    if not rows:
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    content = f"# Papers on {topic.replace('_',' ').title()}\n\nTotal Papers: {len(rows)}\n\n"
    for r in rows:
        content += f"## {r[1]}\n"
        content += f"- **Paper ID**: {r[0]}\n"
        content += f"- **Authors**: {', '.join(r[2])}\n"
        content += f"- **Published**: {r[5]}\n"
        content += f"- **PDF URL**: {r[4]}\n\n"
        content += f"### Summary\n{r[3][:500]}...\n\n---\n\n"
    return content





@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. 

Follow these instructions:
1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
2. For each paper found, extract and organize the following information:
   - Paper title
   - Authors
   - Publication date
   - Brief summary of the key findings
   - Main contributions or innovations
   - Methodologies used
   - Relevance to the topic '{topic}'

3. Provide a comprehensive summary that includes:
   - Overview of the current state of research in '{topic}'
   - Common themes and trends across the papers
   - Key research gaps or areas for future investigation
   - Most impactful or influential papers in this area

4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


@mcp.tool()
async def check_semantic_cache(query: str) -> dict:
    """Check if a sufficiently similar question was already answered against the current saved papers."""
    await asyncio.to_thread(_ensure_index_loaded)
    corpus_version = SemanticCache.corpus_version(_hybrid_index.paper_ids)
    hit = await asyncio.to_thread(_semantic_cache.lookup, query, corpus_version)
    
    if hit:
        return{
            "hit":True,
            "answer":hit["answer"],
            "similarity": hit["similarity"]
        }
    return{"hit":False,"answer":None}

@mcp.tool()
async def store_semantic_cache(query: str, answer: str)-> dict:
    """Store a verified answer in the semantic cache for furture similar questions."""
    corpus_version = SemanticCache.corpus_version(_hybrid_index.paper_ids)
    await asyncio.to_thread(_semantic_cache.store, query, answer, corpus_version)
    return {"stored":True}



@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    # Initialize and run the server
    #mcp.run(transport='stdio')
    mcp.run(transport='sse')


