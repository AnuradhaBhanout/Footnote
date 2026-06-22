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
    all_papers = {}
    if not os.path.exists(PAPER_DIR):
       return all_papers
 
    for topic_dir in os.listdir(PAPER_DIR):
        file_path = os.path.join(PAPER_DIR,topic_dir,"papers_info.json")
        if os.path.isfile(file_path):
            with open(file_path,"r") as f:
                try:
                    topic_papers = json.load(f)
                except json.JSONDecodeError:
                    continue

            all_papers.update(topic_papers)

    return all_papers

# Turn paper into text text chunks