"""tools read by Render/a load balancer, not by the agent or
a human at all"""


import asyncio
from db.db import init_db,get_conn, put_conn
from server.mcp_app import mcp


@mcp.resource("papers://folders")
async def get_available_folders() -> str:
    def _fetchfolder():  
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT topic FROM papers ORDER BY topic;")
            topics = [r[0] for r in cur.fetchall()]
        #put_conn(conn)
        put_conn(conn)
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
        #put_conn(conn)
        put_conn(conn)
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



@mcp.custom_route("/health", methods=["GET","HEAD"])
async def health(request):
    from starlette.responses import JSONResponse
    db_ok = False
    try:
        conn = get_conn()
        conn.cursor().execute("SELECT 1")
        #put_conn(conn)
        put_conn(conn)
        db_ok = True
    except Exception:
        pass
    return JSONResponse({"status": "ok", "db": db_ok})
