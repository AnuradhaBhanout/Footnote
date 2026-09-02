
import os
import psycopg2
import psycopg2.pool

from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv,find_dotenv



_ = load_dotenv(find_dotenv())
DATABASE_URL = os.getenv("DATABASE_URL")
_pool = psycopg2.pool.SimpleConnectionPool(1,5,DATABASE_URL,connect_timeout =10)



def get_conn():
    """Return a psycopg2 connection with pgvectortype registered."""
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        _pool.putconn(conn, close=True)
        conn = _pool.getconn()

    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        conn.rollback()
    return conn

def put_conn(conn):
    _pool.putconn(conn)


def init_db():
    """"Create required table and extensions if they dont exist.
    call once at server 
    """
    conn = psycopg2.connect(DATABASE_URL,connect_timeout=10)
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()

    conn = get_conn()
    with conn:
        with conn.cursor() as cur:
            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            #Rag index  - one row per paper
            cur.execute(""" 
            CREATE TABLE IF NOT EXISTS paper_embeddings(
                    paper_id    TEXT PRIMARY KEY,
                    title       TEXT,
                    text_chunk  TEXT,
                    embedding   vector(384),
                    updated_at  TIMESTAMP DEFAULT NOW()
                );

            """)

            #Approximate Nearest neighbor index for fast similarity search   # using K- mean clustring
            cur.execute("""
                CREATE INDEX IF NOT EXISTS paper_embeddings_ivfflat
                ON paper_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);                          
                """)
            
            # Semantic cache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache(
                        id                 SERIAL PRIMARY KEY,
                        query              TEXT        NOT NULL,
                        answer             TEXT        NOT NULL,
                        embedding          vector(384) NOT NULL,
                        corpus_version     TEXT        NOT NULL,
                        fetched_papers     JSONB       DEFAULT '[]'::jsonb,
                        created_at         TIMESTAMP   DEFAULT NOW()
                        );
                  """)

            #HNSW
            cur.execute("""
                CREATE INDEX IF NOT EXISTS semantic_cache_hnsw ON semantic_cache USING hnsw (embedding vector_cosine_ops);
            """)
            
            # Add papers
            cur.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id    TEXT PRIMARY KEY,
                    topic       TEXT,
                    title       TEXT,
                    authors     JSONB,
                    summary     TEXT,
                    pdf_url     TEXT,
                    published   TEXT,
                    created_at  TIMESTAMP DEFAULT NOW()
                );
            """)

    #conn.close()
    put_conn(conn)

    print("[db] Tables ready.")


