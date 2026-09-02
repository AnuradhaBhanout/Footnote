
import asyncio



from db.rag_index import HybridIndex
from db.semantic_cache import SemanticCache

_hybrid_index = HybridIndex()
_index_loaded = False

_semantic_cache = SemanticCache(model=_hybrid_index.model) # reuse the loaded model




#Its primary purpose is to defer expensive operations (such as loading deep learning models, reading files, or generating vector embeddings) 
# until the exact moment a user performs their first search, rather than doing it when the server boots up.
def _ensure_index_loaded():
    global _index_loaded
    if not _index_loaded:
        _hybrid_index.load_cache_or_build()
        _index_loaded = True




async def request_embed(paper_ids: list[str]):
    asyncio.create_task(asyncio.to_thread(_hybrid_index.embed_specific, paper_ids))