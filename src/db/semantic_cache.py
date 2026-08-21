import os
#import pickle
import hashlib
import time 
import numpy as np
from db.db import get_conn , put_conn
import json



SIMILARITY_THRESHOLD = 0.92

class SemanticCache:
    #def __init__(self,model,cache_file: str = os.path.join("papers","_semantic_cache.pk1")):  // for local calls
    def __init__(self,model):
        self.model = model
       

    def _save(self,entry: dict):
        """ Insert one new cache entry into Postgres."""
        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_cache (query, answer, embedding, corpus_version, fetched_papers)
                    
                    VALUES(%s, %s, %s::vector, %s, %s::jsonb)

                    """,
                    (
                        entry["query"],
                        entry["answer"],
                        entry["embedding"].tolist(),
                        entry["corpus_version"],
                        json.dumps(entry.get("fetched_papers", [])),
                    ),
                )
       # conn.close()
        put_conn(conn)


    @staticmethod
    def corpus_version(paper_ids: list) -> str:
        # creating a fingerprint of id's
        return hashlib.md5(",".join(sorted(paper_ids)).encode()).hexdigest()
    

    def lookup(self, query:str, current_corpus_version: str)-> dict | None:
       # print("LOOKing up for previous queries asked by the user")

        
        # Convert query to list of numbers
        query_vec = self.model.encode([query],convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        

        conn = get_conn()

        try:
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT query, answer, fetched_papers,
                    1 - (embedding <==> %s::vector) AS similarity
                    FROM semantic_cache
                    WHERE corpus_version = %s
                    ORDER BY embedding <==> %s::vector
                    LIMIT 1;
                    """,
                    (query_vec,current_corpus_version,query_vec),
                    )
                row = cur.fetchone()

        finally:
            put_conn(conn)


        if row is None:
           return None

        matched_query, answer, fetched_papers, similarity = row
        if similarity < SIMILARITY_THRESHOLD:
            return None   

        return {"answer": answer,
                "matched_query": matched_query,
                "similarity": float(similarity),
                "fetched_papers": fetched_papers if fetched_papers is not None else [   ],
                }



    def store(self, query: str, answer: str, current_corpus_version: str, fetched_papers: list = None) -> None:

        query_vec = self.model.encode([query],convert_to_numpy = True,normalize_embeddings = True)[0]
        #query_vec = list(self.model.embed([query]))[0]

        
        entry =     {
                "embedding": query_vec,
                "query": query,
                "answer": answer,
                "corpus_version": current_corpus_version, # state of the library
                "fetched_papers": fetched_papers or [],
                "created_at": time.time(),
            }
        
        #########self.entries.append(entry)
        self._save(entry) # Write to disk
