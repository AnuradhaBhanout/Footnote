
"""just, what papers exist and what do we know about them."""


from db.db import init_db,get_conn, put_conn


def load_all_papers() -> dict:
    conn = get_conn()
    papers = {}
    
    # 1. 'with conn:' establishes the transaction block required by server-side cursors
    with conn: 
        with conn.cursor(name="stream_papers_cursor") as cur:
            # 2. Set batch size for streaming network requests
            cur.itersize = 2000 
            cur.execute("SELECT paper_id, title, authors, summary, pdf_url, published FROM papers;")
            
            # 3. Stream each row line-by-line instead of using .fetchall()
            for r in cur:
                papers[r[0]] = {
                    "title": r[1],
                    "authors": r[2],
                    "summary": r[3],
                    "pdf_url": r[4],
                    "published": r[5],
                }
    #conn.close()
    put_conn(conn)
    return papers




def get_papers_fingerprint() -> tuple[int, str]:
    
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), COALESCE(MAX(paper_id), '') FROM papers;")
        row = cur.fetchone()
    #conn.close()
    put_conn(conn)
    return row[0], row[1]

