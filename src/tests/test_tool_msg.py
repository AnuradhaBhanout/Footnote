import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

async def main():
    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(
            stdio_client(StdioServerParameters(command="uv", args=["run", "research_server.py"]))
        )
        session = await stack.enter_async_context(ClientSession(*transport))
        await session.initialize()
        result = await session.call_tool("extract_info", {"paper_id": "2601.18779v1"})
        print(repr(result))

asyncio.run(main())