import os
import time
import json
from dotenv import load_dotenv,find_dotenv
#from anthropic import Anthropic
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types

#from openai import AsyncOpenAI
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from typing import List,Dict,TypedDict
import asyncio
import nest_asyncio

nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from graph_pipeline import build_graph
#from langchain_ollama import ChatOllama
import uuid
import logging
import selectors
from psycopg_pool import AsyncConnectionPool





os.makedirs("logs", exist_ok=True)
# Configure logging to write to debug.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/debug.log"),
        logging.StreamHandler() # This also prints to your terminal
    ]
)
logger = logging.getLogger("RAG-Chatbot")

from  langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

_ = load_dotenv(find_dotenv())

openai_key = os.getenv("OPENAI_API_KEY")
nvidia_key = os.getenv("NVIDIA_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

@tool
def ask_clarification(question:str,options: list[str] | None = None)-> str:
    """Call this INSTEAD of any search  tool when the user's request is ambiguous,
       uses an unclear abbreviation/name, or is missing a detail needed  to act.
       Do NOT call this together with any other tool in the same turn.
       Args:
          questions: A plain-language clarifying question for the user.
          options: 2-4 short possible interpretations to help the user answer quickly.

    """
    options = options or []
    return "CLARIFICATION_REQUESTED"

class MCP_ChatBot:

    def __init__(self):
        
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self.thread_id = str(uuid.uuid4()) 

       
        self.llm = ChatOpenAI(
            model= "gpt-oss-120b",                             #"nvidia/nemotron-3-super-120b-a12b:free",   # Target a solid open-weights model
            openai_api_base="https://api.cerebras.ai/v1",       #"https://openrouter.ai/api/v1", #"https://integrate.api.nvidia.com/v1",     # Connect directly to OpenRouter 
            openai_api_key=os.getenv("CEREBRAS_API_KEY"),       #openai_key,
            max_tokens=2024,
            max_retries=1,
            timeout=33,
            model_kwargs={"parallel_tool_calls": False},
        )

        # self.llm = ChatOpenAI(
        #     model="llama-3.3-70b-versatile",
        #     openai_api_base="https://api.groq.com/openai/v1",
        #     openai_api_key=os.getenv("GROQ_API_KEY"),
        #     max_tokens=2024,
        #     max_retries=1,
        #     timeout=30,
        #     model_kwargs={"parallel_tool_calls": False},
        # )

        # self.llm = ChatOllama(
        #     model="llama3.1",
        #     temperature=0,
        #     #format= "json"
        # )



        self.available_tools = []

        self.available_prompts = []

        self.tool_to_session: Dict[str, ClientSession] = {}

        self.messages = []  
        self._pg_conn = None
        self.app = None
        self.reconnect_event = asyncio.Event()
        self.ready_event = asyncio.Event()

        # NEW
        self._inflight = 0
        self._inflight_lock = asyncio.Lock()
        self._swap_gate = asyncio.Event()
        self._swap_gate.set()   # set = no swap in progress


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
            await self.connect_to_servers()
            await self._build_agent_and_graph()
            self.ready_event.set()
            
            while True:
                await self.reconnect_event.wait()
                self.reconnect_event.clear()
                self.ready_event.clear()
                self._swap_gate.clear()                 # block new acquire_agent() calls
                while self._inflight > 0:                # wait for in-flight requests to release
                    await asyncio.sleep(0.05)
                try:
                    #await self.connect_to_servers()
                    await self._connect_with_retry()
                    await self._rebuild_agent()
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
                self._swap_gate.set()                             # allow new acquire_agent() calls
                self.ready_event.set()
        except asyncio.CancelledError:
            await self.exit_stack.aclose()
            if self._pg_conn:
                #await self._pg_conn.close()
                await self._pg_pool.close()
            raise 
            
    async def _connect_with_retry(self,attempts: int = 4, delay: float=3.0):
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




    async def connect_to_server(self,server_name: str,server_config: dict)-> None:
      """connect to a single MCP server and adapt tools natively into LangChain"""
      try:
           if "url" in server_config:
            # SSE transport
            transport = await self.exit_stack.enter_async_context(
                sse_client(server_config["url"])
            )
           else:
               server_params = StdioServerParameters(**server_config)
               transport = await self.exit_stack.enter_async_context(
               stdio_client(server_params)
           )

           read,write = transport
           session = await self.exit_stack.enter_async_context(
               ClientSession(read,write)
           )

           await session.initialize()
          # self.sessions.append(session)
        
            
           

           try:
            # list of available tools for this session
            response = await session.list_tools()
            tools = response.tools
            for tool_ in tools:
                #self.tool_to_session[tool.name] = session
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
               for prompt  in prompts_response.prompts:
                   self.sessions[prompt.name] = session
                   self.available_prompts.append({"name": prompt.name,"description": prompt.description,"arguments": prompt.arguments})

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
        # tool_names_str =  ", ".join([t.name for t in self.available_tools])

        # self.agent = create_agent(
        #     model = self.llm,
        #     tools=self.available_tools + [ask_clarification],

        #     system_prompt=(
        #     f"SYSTEM ROLE: You are an expert Research Assistant with access to these specific tools: [{tool_names_str}, ask_clarification].\n\n"

        #     "CRITICAL TOOL RULES:\n"
        #     "0. FIRST, judge if the request is clear enough to act on. If it uses a short/ambiguous "
        #     "term, refers to 'that paper' or similar without specifying which, or is missing a needed "
        #     "detail — call ask_clarification with a plain-language question and 2-4 short options. "
        #     "Do NOT call any other tool in the same turn if you call ask_clarification.\n"
        #     "0.5. When calling search_papers or hybrid_search_papers, extract the core topic/title as a "
        #     "clean search phrase — strip filler like 'stands for', 'is about', 'called'. E.g. if the user "
        #     "says 'POPE stands for Privileged On-Policy Exploration', search for 'Privileged On-Policy "
        #     "Exploration', not the full sentence.\n"
        #     "1. NEVER invent a tool name. Use ONLY the names listed above.\n"
        #     "2. For paper info use ONLY: hybrid_search_papers, search_papers, extract_info.\n"
        #     #"3. After calling search_papers, you MUST call extract_info for EACH paper_id returned.\n"
        #     "3. After calling search_papers, you MUST call extract_info ONCE with the list of ALL paper_ids returned.\n"
        #     "4. Only after extract_info calls are complete, write your final summary.\n"
        #     "5. NEVER summarize a paper without first calling extract_info on its paper_id.\n"
        #     "6. When you use a tool, you MUST wait for the tool output before claiming you have finished the task.\n"
        #     "7. If a tool result for 'hybrid_search_papers' has 'evaluator_verdict.sufficient: false', "
        #     "try 'search_papers' ONCE with different terms. If that also returns no useful results, "
        #     "STOP searching and tell the user you couldn't find matching papers — do NOT retry more than once.\n\n"

        #     "CITATION & INTEGRITY RULES:\n"
        #     "- You must use the EXACT title and authors as returned by the tools.\n"
        #     "- NEVER alter, paraphrase, or invent a paper title or finding.\n"
        #     "- If a paper is not relevant to the query, EXCLUDE it entirely.\n\n"

        #     "OUTPUT FORMAT:\n"
        #     "After all tool calls are complete, provide a friendly, plain-language summary. "
        #     "Provide ONLY the final answer. Do NOT include reasoning or internal thoughts."
        #     )                       
        # )
        EXCLUDED_FROM_AGENT = {"extract_info", "check_semantic_cache", "store_semantic_cache"}
        search_tools = [t for t in self.available_tools if t.name not in EXCLUDED_FROM_AGENT]
        tool_names_str = ", ".join([t.name for t in search_tools])

        self.agent = create_agent(
            model=self.llm,
            tools=search_tools + [ask_clarification],

            system_prompt=(
            f"SYSTEM ROLE: You are an expert Research Assistant with access to these specific tools: [{tool_names_str}, ask_clarification].\n\n"

            "CRITICAL TOOL RULES:\n"
            "0. FIRST, judge if the request is clear enough to act on. If it uses a short/ambiguous "
            "term, refers to 'that paper' or similar without specifying which, or is missing a needed "
            "detail — call ask_clarification with a plain-language question and 2-4 short options. "
            "Do NOT call any other tool in the same turn if you call ask_clarification.\n"
            "0.5. When calling search_papers or hybrid_search_papers, extract the core topic/title as a "
            "clean search phrase — strip filler like 'stands for', 'is about', 'called'. E.g. if the user "
            "says 'POPE stands for Privileged On-Policy Exploration', search for 'Privileged On-Policy "
            "Exploration', not the full sentence.\n"
            "1. NEVER invent a tool name. Use ONLY the names listed above.\n"
            "2. For paper info use ONLY: hybrid_search_papers, search_papers.\n"
            "3. Once your search returns paper_ids, STOP calling tools. Full paper details are fetched "
            "automatically after your search — you do not fetch them yourself.\n"
            "4. If a tool result for 'hybrid_search_papers' has 'evaluator_verdict.sufficient: false', "
            "try 'search_papers' ONCE with different terms. If that also returns no useful results, "
            "STOP searching and tell the user you couldn't find matching papers — do NOT retry more than once.\n\n"

            "CITATION & INTEGRITY RULES:\n"
            "- You must use the EXACT title and authors as returned by the tools.\n"
            "- NEVER alter, paraphrase, or invent a paper title or finding.\n"
            "- If a paper is not relevant to the query, EXCLUDE it entirely.\n\n"

            "OUTPUT FORMAT:\n"
            "After your search, provide ONLY a brief note that you're gathering paper details — "
            "the actual final summary will be written in a second step once details are fetched."
            )
        )





    async def _build_agent_and_graph(self):
        await self._rebuild_agent()
        #cache_check = next((t for t in self.available_tools if t.name == "check_semantic_cache"),None)
        #cache_store = next((t for t in self.available_tools if t.name == "store_semantic_cache"),None)

        graph = build_graph(self.llm,self)#,cache_check,cache_store)

        #self._pg_conn = await psycopg.AsyncConnection.connect(DATABASE_URL,autocommit=True)
        self._pg_pool = AsyncConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            open=False,
        )
        await self._pg_pool.open()
        #self.checkpointer = AsyncPostgresSaver(self._pg_conn)
        self.checkpointer = AsyncPostgresSaver(self._pg_pool)

        await self.checkpointer.setup()     #Creates a Langgraph checkpoint tables on first run
        self.app = graph.compile(checkpointer=self.checkpointer)


    async def connect_to_servers(self):

        """Connect to all configured MCP servers."""
        self.sessions = {}                                        #reset state
        self.available_tools = []
        self.available_prompts = []
        self.tool_to_session = {}
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()

        try:
          with open("server_config.json","r") as file:
            data = json.load(file)
            
          for server_name,server_config in data.get("mcpServers",{}).items():
            print(f"{server_name} JSON printed {server_config}")
            await self.connect_to_server(server_name,server_config)
        
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
             


    # Getting resource
    async def get_resource(self,resource_uri):
        session = self.sessions.get(resource_uri)

        # try any paper resource  session
        if not session and resource_uri.startswith("papers://"):
            for uri ,sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break                                                           
        
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)

            else:
                print("No content available.")

        except Exception as e:
            print(f"Error while getting resources: {e}")



    async def list_prompts(self):
        """Lists all available prompts."""
        if not  self.available_prompts:
            print("No prompts avaiable. ")
            return
        
        print("\nAvailable Prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg,'name') else arg.get('name','')
                    print(f" - {arg_name}")

    
    async def execute_prompt(self,prompt_name,args):
        """Execute the prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        try:
            result = await session.get_prompt(prompt_name,arguments = args)
            if result and result.messages:
                prompt_content = result.messages[0].content

            #Extract content from content
            if isinstance(prompt_content,str):
                text = prompt_content
            elif hasattr(prompt_content,'text'):
                text = prompt_content.text
            else:
                text = " ".join(item.text if hasattr(item,'text') else str(item) for item in prompt_content)

            print(f"\nExecuting prompt '{prompt_name}'...")
            await self.process_query(text)
        
        except Exception as e:
            print(f"Error while executing prompt : {e}")


    async def process_query(self, query:str):
        logger.info(f"--- START PROCESS_QUERY: {query} ---")
        #create_agent automates parallel calls, self-corrects minor tool exceptions, 
        # and applies protec tion frameworks against runaway infinite routing loops.
        # tool_names_str =  ", ".join([t.name for t in self.available_tools])
        
        # agent = create_agent(
        #     model = self.llm,
        #     tools=self.available_tools,
        #     system_prompt=(
        #     f"SYSTEM ROLE: You are an expert Research Assistant with access to these specific tools: [{tool_names_str}].\n\n"
            
        #     "CRITICAL TOOL RULES:\n"
        #     "1. NEVER invent a tool name. Use ONLY the names listed above.\n"
        #     "2. When you use a tool, you MUST wait for the tool output before claiming you have finished the task.\n"
        #     "3. If a tool result for 'hybrid_search_papers' has 'evaluator_verdict.sufficient: false', "
        #     "do NOT provide an answer. Instead, try 'search_papers' or 'fetch' to find better information.\n\n"

        #     "CITATION & INTEGRITY RULES:\n"
        #     "- You must use the EXACT title and authors as returned by the tools.\n"
        #     "- NEVER alter, paraphrase, or invent a paper title or finding.\n"
        #     "- If a paper is not relevant to the query, EXCLUDE it entirely.\n\n"

        #     "OUTPUT FORMAT:\n"
        #     "After all tool calls are complete, provide a friendly, plain-language summary of your findings. "
        #     "If citations are used, list them clearly."
        #     )                       
        # )
        
        # cache_check = next(t for t in self.available_tools if t.name == "check_semantic_cache")
        # cache_store = next(t for t in self.available_tools if t.name == "store_semantic_cache")

        # graph = build_graph(self.llm,agent,cache_check,cache_store)

        # async with AsyncSqliteSaver.from_conn_string("conversations.db") as checkpointer:
        #     app = graph.compile(checkpointer= checkpointer)
        config = {"configurable":
                {
                    "thread_id": self.thread_id
                }}
        

        logger.info("Invoking Graph...")
        result = await self.app.ainvoke(
            {
                "original_query":query,
                "current_query": query,
                "messages":self.messages,
                "retry_count":0,
                "clarification_question": None,   
                "clarification_options": [],  
            },
            config
        )

        while "__interrupt__" in result:
            logger.info("Graph Interrupted: Waiting for user clarification.")
            question_data = result["__interrupt__"][0].value
            
            if isinstance(question_data, dict):
                print(f"AI: {question_data.get('question', 'Could you clarify?')}")
                if question_data.get("options"):
                    print("Possible meanings:", ", ".join(question_data["options"]))
            else:
                print(f"AI: {question_data}")

            
            #Get human clarification
            answer = (await asyncio.to_thread(input,"\nQuery:")).strip()

            # Resume graph execution with the user's answer
            logger.info(f"Resuming graph with: {answer}")
            result = await self.app.ainvoke(Command(resume=answer),config)

        all_messages = result.get("messages",self.messages)
        # Keep only human/AI conversation turns — no tool call noise
        self.messages = [
            m for m in all_messages
            if isinstance(m, (HumanMessage, AIMessage))
            and not getattr(m, "tool_calls", None)
        ]
        self.messages = self.messages[-20:]                  # keep last 20 messages only


        final_response = result.get('draft_answer')
        logger.info("Graph finished execution.")
        
        if final_response:
            print(f"\nAI: {final_response}")

            

 #########   this part is now calling vai run_agent node in graph_pipeline.py   #################

        # # messages = [{'role':'user','content':'query'}]
        # # replaced with LANGCHAIN
        # self.messages.append(HumanMessage(content=query))

        t0 = time.time()
        # # replaced with LANGCHAIN
        # # llm_with_tools = self.llm.bind_tools(self.available_tools)
        # # response = await llm_with_tools.ainvoke(self.messages)
        # agent_state = await agent.ainvoke({"messages": self.messages})
        print(f"[timing] agent response took {time.time() - t0:.2f}s")

        # self.messages = agent_state["messages"]

        # # self.messages.append(response)

        # # if response.content:
        # #     print(f"AI: {response.content}")

        # final_response = self.messages[-1]
        # if final_response.content:
        #     print(f"AI: {final_response.content}")

            

                   


    async def chat_loop(self):
        """ Run an interactive chat loop"""
        print("\n MCP Chatbot Started !!!!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = (await asyncio.to_thread(input,"\nQuery: ")).strip()

                if query.lower() == "quit":
                    break

                if not query:
                    continue

                # Check for resource syntax first
                if query.startswith('@'):
                    #Remove @ sign
                    topic = query[1:]
                    if topic == "folders":
                       resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()

                    if command == '/clear':
                        self.messages = []
                        print("Conversation history cleared.")
                        continue

                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts)<2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue

                        prompt_name =parts[1]
                        args = {}

                        for arg in parts[2:]:
                            if '=' in arg:
                                key,value = arg.split('=',1)
                                args[key]= value

                        await self.execute_prompt(prompt_name,args)
                    else:
                        print(f"Unknown command: {command}")
                    continue




                await self.process_query(query)
                print("\n")
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                #print(f"\nError encountered during chat_loop : {str(e)}")

    

    async def cleanup(self):
        """Cleanly close all resources """
        if hasattr(self, '_pg_conn'):
            #await self._pg_conn.close()      #Close Postgres connection
            await self._pg_pool.close()
        await self.exit_stack.aclose()   # Close MCP server subprocess


                
async def main():
    print("CHATBOT")
    chatbot = MCP_ChatBot()
    try:
        t0 = time.time()
        #await chatbot.connect_to_servers() 
        await chatbot._connect_with_retry()
        print(f"[timing] server connection took {time.time() - t0:.2f}s")
        await chatbot._build_agent_and_graph()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() 
  

if __name__ == "__main__":
    asyncio.set_event_loop_policy(
        asyncio.DefaultEventLoopPolicy()
    )
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())