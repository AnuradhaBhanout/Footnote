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