import os
import json
#import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

from db import get_conn 


PAPER_DIR = "papers"
#INDEX_CACHE = os.path.join(PAPER_DIR,"_rag_index.pk1")

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
def paper_to_text(paper_info:dict)-> str:
    """Concatenate title and summary into one chunk for both  BM25 and embeddings."""
    title = paper_info.get("title","")
    summary = paper_info.get("summary","")
    return f"{title}. {summary}"

# BM25 just need a lowercase tokens (Tokerizer for BM25)
def simple_tokenize(text: str)-> list:
    return text.lower().split()

class HybridIndex:
    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.paper_ids = []
        self.texts = []
        self.embeddings = None
        self.bm25 = None
        
    def build(self):
        """Recompute the index from whatever is currently in papers/."""
        papers = load_all_papers()
        self.paper_ids = list(papers.keys())
        self.texts = [paper_to_text(papers[pid]) for pid in self.paper_ids]

        #If no papers have been downloaded yet (or the database is empty),   
        if not self.texts:
            self.embeddings = np.zeros((0,384))
            self.bm25 = BM25Okapi([[""]])
            return
        
        #Dense vector - one per paper
        self.embeddings = self.model.encode(
            self.texts,convert_to_numpy=True,normalize_embeddings=True
        )

        #Sparse index - needs tokenized corpus
        tokenized_corpus = [simple_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._upsert_to_db(papers)

    
    def _upsert_to_db(self,papers: dict):
        """
        Upsert paper embeddings into paper_embeddings table.
        ON CONFLICT updates embedding + text_chunk in case summary changed.
        
        """
        rows = [
            (
                pid,
                papers[pid].get("title",""),
                self.texts[i],
                self.embeddings[i].tolist(),         #pgvector expects a list     
            )
            for i ,pid in enumerate(self.paper_ids)
        ]

        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO paper_embeddings(paper_id,title,text_chunk,embedding)
                    VALUES %s
                    ON CONFLICT (paper_id) DO UPDATE
                        SET title      = EXCLUDED.title,
                            text_chunk = EXCLUDED.text_chunk,
                            embedding  = EXCLUDED.embedding,
                            updated_at = NOW()
                    """,
                    rows,
                    template="(%s, %s, %s, %s::vector)",
                )
        conn.close()


    def load_cache_or_build(self):
        """
        Load embeddings from Postgres.
        Falls back to build() if the table is empty.
        """
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT paper_id,title,text_chunk,embedding FROM paper_embeddings;")
            rows = cur.fetchall()
        conn.close()

        if not rows:
            #Nothing in DB yet - build from disk
            self.build()
            return
        
        self.paper_ids = [r[0] for r in rows]
        self.texts = [r[2] for r in rows]

        #pgvectors returns embedding as Python lists: convert to numpy
        self.embeddings = np.array([r[3] for r in rows],dtype = np.float32)
        self.bm25 = BM25Okapi( [simple_tokenize(t) for t in self.texts])

    # # Cache to disk 
    # def _save_cache(self):
    #     with open(INDEX_CACHE,"wb") as f:
    #         pickle.dump(
    #             {
    #                 "paper_ids":self.paper_ids,"texts":self.texts,"embeddings":self.embeddings},f,
                
    #         )

    # def load_cache_or_build(self):
    #     if os.path.isfile(INDEX_CACHE):
    #         with open(INDEX_CACHE,"rb") as f:
    #             data = pickle.load(f)
    #         self.paper_ids = data["paper_ids"]
    #         self.texts = data["texts"]
    #         self.embeddings = data["embeddings"]
    #         tokenized_corpus = [simple_tokenize(t) for t in self.texts]
    #         self.bm25 = BM25Okapi(tokenized_corpus) if self.texts else BM25Okapi([[""]])

    #     else:
    #         self.build()

    def refresh_if_stale(self):
        """
        Compare paper_ids on disk vs in memory.
        """
        current_papers = load_all_papers()
        if set(current_papers.keys()) != set(self.paper_ids):
            self.build()



# Defines the entry point for the hybrid search query
    def search(self,query: str,top_k: int=5,alpha: float =0.5) -> list:

       if not self.paper_ids:
          return[]
    
       # used to generate semantic text embeddings  
       query_vec = self.model.encode([query] ,convert_to_numpy=True,normalize_embeddings=True)[0]  
    
       dense_scores = self.embeddings @ query_vec   #Cosine Similarity

       # Cal Sparse(Keyword) score
       bm25_scores = np.array(self.bm25.get_scores(simple_tokenize(query)))

       dense_norm = self._min_max(dense_scores) # 0 or 1

       bm25_norm = self._min_max(bm25_scores)   # 0 or 1

       combined = alpha * dense_norm + (1-alpha) * bm25_norm

       top_indices = np.argsort(combined)[::-1][:top_k]

       return[
           {
               "paper_id":self.paper_ids[i],
               "score":float(combined[i]),
               "dense_score": float(dense_norm[i]),
               "bm25_score":float(bm25_norm[i]),
               
           }
           for i in top_indices
       ]


    @staticmethod
    def _min_max(scores: np.ndarray)-> np.ndarray:
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min())/(scores.max() - scores.min()) 




