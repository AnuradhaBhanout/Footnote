import asyncio
from arq.connections import RedisSettings
import os
from server.index_state import _hybrid_index


async def embed_papers(ctx,paper_ids: list[str]):
    await asyncio.to_thread(_hybrid_index.embed_specific,paper_ids )

class WorkerSettings:
    functions = [embed_papers]
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL"))