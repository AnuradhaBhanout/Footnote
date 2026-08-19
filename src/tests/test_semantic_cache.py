"""
Run from src/: uv run test_semantic_cache.py
Uses the REAL SentenceTransformer (needs network on first run to download the model).
"""
from sentence_transformers import SentenceTransformer
from db.semantic_cache import SemanticCache
import tempfile

model = SentenceTransformer("all-MiniLM-L6-v2")
cache = SemanticCache(model=model, cache_file=tempfile.mktemp(suffix=".pkl"))

v1 = SemanticCache.corpus_version(["paper_a", "paper_b", "paper_c"])

print("--- Test 1: store then look up a similar phrasing ---")
cache.store("how does sourdough fermentation work", "ANSWER ABOUT FERMENTATION", v1)
hit = cache.lookup("explain how sourdough fermentation works", v1)
print("Hit:", hit)
assert hit is not None, "FAILED: expected a hit on similar phrasing"
print("PASSED\n")

print("--- Test 2: unrelated query should miss ---")
miss = cache.lookup("what is the capital of France", v1)
print("Result:", miss)
assert miss is None, "FAILED: expected a miss on unrelated query"
print("PASSED\n")

print("--- Test 3: corpus changed should invalidate, even with identical text ---")
v2 = SemanticCache.corpus_version(["paper_a", "paper_b", "paper_c", "paper_d"])
miss2 = cache.lookup("how does sourdough fermentation work", v2)
print("Result:", miss2)
assert miss2 is None, "FAILED: expected invalidation after corpus change"
print("PASSED\n")

print("All semantic_cache tests passed.")
