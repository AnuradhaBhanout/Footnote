import os
import json
#import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from db.paper_store import load_all_papers,get_papers_fingerprint
from psycopg2.extras import execute_values
from db.embedding_model import EmbeddingModel

from db.db import get_conn ,put_conn


PAPER_DIR = "papers"






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
    def __init__(self,model_name:str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = EmbeddingModel(model_name=model_name)
        self.paper_ids = []
        self.texts = []
        self.titles = []
        self.embeddings = None
        self.bm25 = None
        self._last_fingerprint = None


    def build(self):
        """Recompute the index from whatever is currently in papers/."""
        papers = load_all_papers()
        self.paper_ids = list(papers.keys())
        self.texts = [paper_to_text(papers[pid]) for pid in self.paper_ids]
        self.titles = [papers[pid].get("title","Unknown")  for pid in self.paper_ids]

        #If no papers have been downloaded yet (or the database is empty),   
        if not self.texts:
            self.embeddings = np.zeros((0,384))
            self.bm25 = BM25Okapi([[""]])
            return
        
        #Dense vector - one per paper
        self.embeddings = self.model.encode(
            self.texts,convert_to_numpy=True,normalize_embeddings=True,batch_size=8, show_progress_bar=False
        )

        #self.embeddings = np.array(list(self.model.embed(self.texts)))

        #Sparse index - needs tokenized corpus
        tokenized_corpus = [simple_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._upsert_to_db(paper_ids=self.paper_ids, titles=self.titles,texts=self.texts,embeddings=self.embeddings)

    
    def _upsert_to_db(self,*,paper_ids,titles,texts,embeddings):
        """
        Upsert paper embeddings into paper_embeddings table.
        ON CONFLICT updates embedding + text_chunk in case summary changed.
        
        """
        rows = [
            (
                pid,
                titles[i],
                texts[i],
                embeddings[i].tolist(),         #pgvector expects a list     
            )
            for i ,pid in enumerate(paper_ids)
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
        #conn.close()
        put_conn(conn)


    def load_cache_or_build(self):
        """
        Load embeddings from Postgres.
        Falls back to build() if the table is empty.
        """
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT paper_id,title,text_chunk,embedding FROM paper_embeddings;")
            rows = cur.fetchall()
        #conn.close()
        put_conn(conn)

        if not rows:
            #Nothing in DB yet - build from disk
            self.build()
            return
        
        self.paper_ids = [r[0] for r in rows]
        self.texts = [r[2] for r in rows]
        self.titles = [r[1] for r in rows]

        #pgvectors returns embedding as Python lists: convert to numpy
        self.embeddings = np.array([r[3] for r in rows],dtype = np.float32)
        self.bm25 = BM25Okapi( [simple_tokenize(t) for t in self.texts])


 

    def refresh_if_stale(self):
        """
        Compare paper_ids on disk vs in memory.
        """
        fingerprint = get_papers_fingerprint()
        if fingerprint != self._last_fingerprint:
            self.build()
            self._last_fingerprint = fingerprint




# Defines the entry point for the hybrid search query
    def search(self,query: str,top_k: int=5,alpha: float =0.5) -> list:

       if not self.paper_ids:
          return[]
    
       # used to generate semantic text embeddings  
       query_vec = self.model.encode([query] ,convert_to_numpy=True,normalize_embeddings=True)[0]  
       #query_vec = list(self.model.embed([query]))[0]
    
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
               "title": self.titles[i],
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


    def embed_specific(self,paper_ids:list[str]):
        conn = get_conn()
        with conn.cursor() as cur:

            cur.execute("SELECT paper_id FROM paper_embeddings WHERE paper_id = ANY(%s)",(paper_ids,)),
            already_done = {r[0] for r in cur.fetchall()}
            remaining = [pid for pid in paper_ids if pid not in already_done]

            if not remaining:
                put_conn(conn)
                return



            cur.execute(
                "SELECT paper_id,title, authors,summary,pdf_url, published FROM papers WHERE paper_id = ANY(%s)",
                (remaining,)
            )
            rows = cur.fetchall()

        put_conn(conn)
        texts = [paper_to_text({"title":r[1],"summary":r[3]}) for r in rows]
        titles = [r[1] for r in rows]
        ids = [r[0] for r in rows]

        embeddings = self.model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,batch_size=8,show_progress_bar=False)
        self._upsert_to_db(paper_ids=ids, titles=titles,texts=texts, embeddings=embeddings)
