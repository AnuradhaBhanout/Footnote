from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from db.db import get_conn, put_conn
from db.rag_index import HybridIndex

conn = get_conn()
try:
    with conn.cursor() as cur:
        cur.execute("SELECT paper_id FROM papers")
        ids = [r[0] for r in cur.fetchall()]
finally:
    put_conn(conn)

assert ids, "no papers found - check DATABASE_URL points at Neon"
print(f"{len(ids)} papers in corpus")
HybridIndex().embed_specific(ids)
print("done")