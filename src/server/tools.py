"""@mcp.tool()-decorated function the LangChain agent 
 can invoke."""

import asyncio
import json
import logging
import os
import sys

from typing import List
import arxiv

from db.citation_verifier import _title_overlap_ration
from db.db import init_db,get_conn, put_conn
from db.rag_index import load_all_papers
from db.semantic_cache import SemanticCache

from server.index_state import _hybrid_index,_semantic_cache,_ensure_index_loaded
from server.mcp_app import mcp

from server.relevance import evaluate_relevance,_quoted_phrase

PAPER_DIR = "papers"







#Call LLM for dynamic search across all saved papers on disk using hybrid search architecture.
@mcp.tool()
async def hybrid_search_papers(query: str,top_k: int = 5,alpha: float = 0.5)-> dict:
    
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
    ###ADDED##
    quoted = _quoted_phrase(query)
    if quoted and judgment.get("sufficient"):
        best = next((r for r in results if r["paper_id"] == judgment.get("best_paper_id")), None)
        title_match = _title_overlap_ration(quoted, best["title"]) if best else 0.0
        if title_match < 0.7:
            judgment = {
                "sufficient": False,
                "best_paper_id": None,
                "reason": f"Query names a specific title (\"{quoted}\") not found among retrieved papers (best overlap={title_match:.2f}).",
            }
            ####
    logging.info(f"[evaluator] sufficient={judgment.get('sufficient')} reason={judgment.get('reason')}")

    return {"results": results,"evaluator_verdict": judgment}  







@mcp.tool()
async def search_papers(topic: str, max_results: int = 5) -> dict:                        # List[str]:
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
      papers = await asyncio.wait_for( asyncio.to_thread(lambda: list(client.results(search))), timeout=10.0)
    except asyncio.TimeoutError:
        logging.error(f"search_papers: arxiv timed out for topic '{topic}'")
        return {"paper_ids": [], "sufficient": False, "reason": "arXiv search timed out."}   # was: return []
       # return []
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

  

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
    #return paper_ids
    if not paper_ids:
        return {"paper_ids": [], "sufficient": False, "reason": "arXiv returned no results."}

    await asyncio.to_thread(_ensure_index_loaded)
    texts = [f"{papers_info[pid]['title']}. {papers_info[pid]['summary']}" for pid in paper_ids]
    query_vec = _hybrid_index.model.encode([topic], convert_to_numpy=True, normalize_embeddings=True)[0]
    doc_vecs  = _hybrid_index.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = doc_vecs @ query_vec

    RELEVANCE_FLOOR = 0.35
    kept = [pid for pid, sim in zip(paper_ids, sims) if sim >= RELEVANCE_FLOOR]

    return {"paper_ids": kept, "sufficient": len(kept) > 0,
            "reason": f"{len(kept)}/{len(paper_ids)} arXiv results passed similarity floor {RELEVANCE_FLOOR}."}





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
    put_conn(conn)







@mcp.tool()
async def extract_info(paper_ids: List[str]) -> str:
    if isinstance(paper_ids, str):        # tolerate a model that forgets to wrap in a list
        paper_ids = [paper_ids]

    def _fetch():
        if not paper_ids:
            return []
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT paper_id, title, authors, summary, pdf_url, published "
                "FROM papers WHERE paper_id = ANY(%s)",
                (paper_ids,)
            )
            rows = cur.fetchall()
        put_conn(conn)
        return rows

    rows = await asyncio.to_thread(_fetch)
    found_ids = {r[0] for r in rows}
    not_found = [pid for pid in paper_ids if pid not in found_ids]

    papers_list = [
        {
            "paper_id": r[0],
            "title": r[1],
            "authors": r[2],
            "summary": r[3],
            "pdf_url": r[4],
            "published": r[5],
        }
        for r in rows
    ]

    return json.dumps({"papers": papers_list, "not_found": not_found}, indent=2)



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
            "similarity": hit["similarity"],
            "fetched_papers": hit.get("fetched_papers", []),
        }
    return{"hit":False,"answer":None,"fetched_papers":[]}




@mcp.tool()
async def store_semantic_cache(query: str, answer: str, fetched_papers: List[dict] = None)-> dict:
    """Store a verified answer in the semantic cache for furture similar questions."""
    corpus_version = SemanticCache.corpus_version(_hybrid_index.paper_ids)
    await asyncio.to_thread(_semantic_cache.store, query, answer, corpus_version, fetched_papers)
    return {"stored":True}
