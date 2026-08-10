import os
#import pickle
import hashlib
import time 
import numpy as np
from db import get_conn
import json



SIMILARITY_THRESHOLD = 0.92

class SemanticCache:
    #def __init__(self,model,cache_file: str = os.path.join("papers","_semantic_cache.pk1")):  // for local calls
    def __init__(self,model):
        self.model = model
       # self.cache_file = cache_file
        self.entries: list = []  # this will hold cached answers in memory
        self._load()


    # def _load(self):                               //for local calls
    #     if os.path.isfile(self.cache_file):
    #         with open(self.cache_file,"rb") as f:
    #            # 'pickle' converts the python list into a file
    #            self.entries = pickle.load(f)
    
    # def _save(self):                              //for local calls
    #     with open(self.cache_file,"wb") as f:
    #         pickle.dump(self.entries,f)



    def _load(self):
        """ Load all cache entries from Postgres into memory at startup."""
        try:
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                            SELECT query, answer, embedding, corpus_version, fetched_papers, created_at
                            FROM semantic_cache ORDER BY created_at ASC;
                            """)
                rows = cur.fetchall()
            conn.close()

            self.entries = [
                {
                    "query":  r[0],
                    "answer": r[1],
                    "embedding": np.array(r[2],dtype = np.float32),
                    "corpus_version": r[3],
                    "fetched_papers": r[4] if r[4] is not None else [],
                    "created_at": r[5].timestamp() if r[5] else time.time(),
                }
                for r in rows
            ]
        
        except Exception as e:
            #Dont crash on startup if db is not ready yet.
            print(f"[semantic_cache] Warning: could not load from DB: {e}") 
            self.entries = []


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
        conn.close()


    @staticmethod
    def corpus_version(paper_ids: list) -> str:
        # creating a fingerprint of id's
        return hashlib.md5(",".join(sorted(paper_ids)).encode()).hexdigest()
    

    def lookup(self, query:str, current_corpus_version: str)-> dict | None:
       # print("LOOKing up for previous queries asked by the user")
        #self._load()
        if not self.entries:
            return None
        
        # Convert query to list of numbers
        query_vec = self.model.encode([query],convert_to_numpy=True, normalize_embeddings=True)[0]
        #query_vec = list(self.model.embed([query]))[0]

        best_score = -1.0
        best_entry = None

        for entry in self.entries:

            if entry["corpus_version"] != current_corpus_version:
                continue

            # calculating similarity between query_vec and  cached vec
            score = float(np.dot(entry["embedding"],query_vec))

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
                return {"answer": best_entry["answer"],
                        "matched_query": best_entry["query"],
                        "similarity": best_score,
                        "fetched_papers": best_entry.get("fetched_papers", []),
                        }

        return None


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
        
        self.entries.append(entry)
        self._save(entry) # Write to disk
