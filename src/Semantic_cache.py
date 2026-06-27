import os
import pickle
import hashlib
import time
import numpy as np


SIMILARITY_THRESHOLD = 0.92

class SemanticCache:
    def __init__(self,model,cache_file: str = os.path.join("papers","_semantic_cache.pk1")):
        self.model = model
        self.cache_file = cache_file
        self.entries: list = []  # this will hold cached answers in memory
        self._load()


    def _load(self):
        if os.path.isfile(self.cache_file):
            with open(self.cache_file,"rb") as f:
               # 'pickle' converts the python list into a file
               self.entries = pickle.load(f)
    
    def _save(self):
        with open(self.cache_file,"wb") as f:
            pickle.dump(self.entries,f)
        
    @staticmethod
    def corpus_version(paper_ids: list) -> str:
        # creating a fingerprint of id's
        return hashlib.md5(",".join(sorted(paper_ids)).encode()).hexdigest()
    

    def lookup(self, query:str, current_corpus_version: str)-> dict | None:
        
        if not self.entries:
            return None
        
        # Convert query to list of numbers
        query_vec = self.model.encode([query],convert_to_numpy=True, normalize_embeddings=True)[0]

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
                return {"answer": best_entry["answer"],"matched_query": best_entry["query"],"similarity": best_score}

            return None


    def store(self, query: str, answer: str, current_corpus_version: str) -> None:

        query_vec = self.model.encode([query],convert_to_numpy = True,normalize_embeddings = True)[0]

        self.entries.append(
            {
                "embedding": query_vec,
                "query": query,
                "answer": answer,
                "corpus_version": current_corpus_version, # state of the library
                "created_at": time.time(),
            }
        )

        self._save() # Write to disk
