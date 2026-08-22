from arq import create_pool
from arq.connections import RedisSettings
import os

_redis_pool  = None


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


async def get_redis_pool():
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_URL")))
    return _redis_pool

async def request_embed(paper_ids:list[str]):
    pool = await get_redis_pool()
    await pool.enqueue_job("embed_papers",paper_ids)