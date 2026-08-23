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
from server import tools, resources_and_prompts


if __name__ == "__main__":
    import asyncio
    import uvicorn
    from arq.worker import Worker
    from server.worker import WorkerSettings

    app = mcp.sse_app()

    async def start_embed_worker():
        worker = Worker(
            functions=WorkerSettings.functions,
            redis_settings = WorkerSettings.redis_settings,
            handle_signals=False,
        )

        app.state.embed_worker =  worker
        app.state.embed_worker_task = asyncio.create_task(worker.async_run())

    async def stop_embed_worker():
        task = getattr(app.state,"embed_worker_task",None)
        worker = getattr(app.state,"embed_worker",None)

        if task:
            task.cancel()
        if worker:
            await worker.close()

    app.add_event_handler("startup",start_embed_worker)
    app.add_event_handler("shutdown",stop_embed_worker)

    uvicorn.run(app, host="0.0.0.0",port=int(os.environ.get("PORT") or 8001))


