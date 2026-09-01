import logging
import os


from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


from db.db import init_db


init_db()               # creates tables on first run, safe to call every time


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/research_server_debug.log"),
        logging.StreamHandler() # This also prints to your terminal
    ]
)



from server .mcp_app import mcp
from server import tools


if __name__ == "__main__":
    import asyncio
    import uvicorn
    from arq.worker import Worker
    from server.worker import WorkerSettings
    from contextlib import asynccontextmanager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    mcp_app = mcp.sse_app()

    @asynccontextmanager
    async def lifespan(app):
        worker = Worker(
            functions=WorkerSettings.functions,
            redis_settings = WorkerSettings.redis_settings,
            handle_signals=False,
        )       

        worker_task = asyncio.create_task(worker.async_run())

        try:
            yield
        finally:
            worker_task.cancel()
            await worker.close()

    app = Starlette(routes=[Mount("/", app=mcp_app)], lifespan=lifespan)

    uvicorn.run(app, host="0.0.0.0",port=int(os.environ.get("PORT") or 8001))


