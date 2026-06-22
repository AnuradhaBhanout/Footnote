import os
import json
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PAPER_DIR = "papers"
INDEX_CACHE = os.path.join(PAPER_DIR,"_rag_index.pk1")

def load_all_papers() -> dict:
    """Walk every topic folder under papers / and merge all papers_info.json files"""
