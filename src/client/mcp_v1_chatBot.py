import asyncio

import os
import uuid
from contextlib import AsyncExitStack

from mcp import ClientSession
from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from mcp.client.sse import sse_client

from psycopg_pool import AsyncConnectionPool
 
from client.agent_prompt import build_system_prompt, filter_search_tools
from client.tools import ask_clarification
from graph.graph_pipeline import build_graph
from log_setup import setup_logging
 

 
logger = setup_logging("RAG-Chatbot", "debug.log")
 
_ = load_dotenv(find_dotenv())
 
DATABASE_URL = os.getenv("DATABASE_URL")
 
 
class MCP_ChatBot:
 
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self.thread_id = str(uuid.uuid4())
 
        self.llm = ChatOpenAI(
            model="gpt-oss-120b",
            openai_api_base="https://api.cerebras.ai/v1",
            openai_api_key=os.getenv("CEREBRAS_API_KEY"),
            max_tokens=2024,
            max_retries=1,
            timeout=10,
            model_kwargs={"parallel_tool_calls": False, "reasoning_effort": "low"},
        )
 
        self.available_tools = []
        self.available_prompts = []
 
        self.messages = []
        self._pg_conn = None
        self.app = None
        self.reconnect_event = asyncio.Event()
        self.ready_event = asyncio.Event()
 
        self._inflight = 0
        self._inflight_lock = asyncio.Lock()
        self._swap_gate = asyncio.Event()
        self._swap_gate.set()  # set = no swap in progress
 
    async def acquire_agent(self):
        while True:
            await self._swap_gate.wait()
            async with self._inflight_lock:
                if self._swap_gate.is_set():
                    self._inflight += 1
                    return self.agent
            # a swap started between wait() and lock — loop and retry
 
    async def release_agent(self):
        async with self._inflight_lock:
            self._inflight -= 1


    async def session_manager(self):
        try:
            await self._connect_with_retry()
            await self._build_agent_and_graph()
            self.ready_event.set()
            #asyncio.create_task(self._keepalive())
 
            while True:
                await self.reconnect_event.wait()
                self.reconnect_event.clear()
                self.ready_event.clear()
                self._swap_gate.clear()  # block new acquire_agent() calls
                while self._inflight > 0:  # wait for in-flight requests to release
                    await asyncio.sleep(0.05)
                try:
                    await self._connect_with_retry()
                    await self._rebuild_agent()
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
                self._swap_gate.set()  # allow new acquire_agent() calls
                self.ready_event.set()
        except asyncio.CancelledError:
            await self.exit_stack.aclose()
            if self._pg_conn:
                await self._pg_pool.close()
            raise
 
    async def _connect_with_retry(self, attempts: int = 4, delay: float = 3.0):
        last_err = None
        for i in range(attempts):
            try:
                await self.connect_to_servers()
                if self.available_tools:
                    return
                last_err = RuntimeError("connected but zero tool registered")
 
            except Exception as e:
                last_err = e
            logger.warning(f"connection attempt{i+1}/{attempts} failed: {last_err}. Retrying in {delay}s")
            await asyncio.sleep(delay)
        raise last_err
 
    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """connect to a single MCP server and adapt tools natively into LangChain"""
        try:
            transport = await self.exit_stack.enter_async_context(
                sse_client(server_config["url"])
            )
 
            read, write = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
 
            await session.initialize()
 
            try:
                # list of available tools for this session
                response = await session.list_tools()
                tools = response.tools
                for tool_ in tools:
                    self.sessions[tool_.name] = session
 
                raw_langchain_tools = [
                    convert_mcp_tool_to_langchain_tool(session, tool_) for tool_ in response.tools
                ]
                for lc_tool in raw_langchain_tools:
                    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
                        schema = lc_tool.args_schema
                        if isinstance(schema, dict):
                            schema_dict = schema
                        elif hasattr(schema, "model_dump"):
                            schema_dict = schema.model_dump()
                        else:
                            schema_dict = None
                        if schema_dict is not None:
                            schema_dict.pop("title", None)
                            for prop in schema_dict.get("properties", {}).values():
                                if isinstance(prop, dict):
                                    prop.pop("title", None)
                            lc_tool.args_schema = schema_dict
 
                        lc_tool.description = (lc_tool.description or "").strip()
 
                    self.available_tools.append(lc_tool)
                logger.info(f"[SCHEMA DUMP] " + "; ".join(f"{t.name}: {getattr(t, 'args_schema', None)}" for t in raw_langchain_tools))
 
                print("\nConnected to server with tools:", [t.name for t in self.available_tools])
 
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({"name": prompt.name, "description": prompt.description, "arguments": prompt.arguments})
 
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
 
            except Exception as e:
                print(f"Error {e}")
 
        except Exception as e:
            print(f"failed to connect to {server_name}: {e}")
            raise
 
    async def _rebuild_agent(self):
        search_tools = filter_search_tools(self.available_tools)
        tool_names_str = ", ".join(t.name for t in search_tools)
 
        self.agent = create_agent(
            model=self.llm,
            tools=search_tools + [ask_clarification],
            system_prompt=build_system_prompt(tool_names_str),
        )
 
    async def _build_agent_and_graph(self):
        await self._rebuild_agent()
 
        graph = build_graph(self.llm, self)
 
        self._pg_pool = AsyncConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            open=False,
        )
        await self._pg_pool.open()
        self.checkpointer = AsyncPostgresSaver(self._pg_pool)
 
        await self.checkpointer.setup()  # Creates a Langgraph checkpoint tables on first run
        self.app = graph.compile(checkpointer=self.checkpointer)
 
    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        self.sessions = {}  # reset state
        self.available_tools = []
        self.available_prompts = []
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()
 
        try:
            port = int(os.environ.get("PORT") or 8000)
            url = os.getenv("MCP_URL") or f"http://127.0.0.1:{port}/mcp/sse"
            await self.connect_to_server("research", {"url": url})
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            raise
 
    async def cleanup(self):
        """Cleanly close all resources """
        if hasattr(self, '_pg_conn'):
            await self._pg_pool.close()
        await self.exit_stack.aclose()  # Close MCP server subprocess