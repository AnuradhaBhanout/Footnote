
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv,find_dotenv


_ = load_dotenv(find_dotenv())
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    """Return a psycopg2 connection with pgvectortype registered."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn
