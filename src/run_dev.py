import sys
import asyncio
import uvicorn

config = uvicorn.Config("api.api:app", host="127.0.0.1", port=8000)
server = uvicorn.Server(config)

if sys.platform == "win32":
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
else:
    server.run()