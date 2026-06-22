import os
import time
import json
from dotenv import load_dotenv,find_dotenv
from anthropic import Anthropic
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types

from openai import AsyncOpenAI
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from typing import List,Dict,TypedDict
import asyncio
import nest_asyncio

nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain.agents import create_agent

from  langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

_ = load_dotenv(find_dotenv())

openai_key = os.getenv("OPENAI_API_KEY")

# class ToolDefinition(TypedDict):
#     name: str
#     description: str
#     input_schema: dict


class MCP_ChatBot:

    def __init__(self):
        
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

       
        self.llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free", # Target a solid open-weights model
            openai_api_base="https://openrouter.ai/api/v1",  # Connect directly to OpenRouter 
            openai_api_key=openai_key,
            max_tokens=2024
        )
        
        self.available_tools = []

        self.available_prompts = []

        self.tool_to_session: Dict[str, ClientSession] = {}

        self.messages = []  

    async def connect_to_server(self,server_name: str,server_config: dict)-> None:
      """connect to a single MCP server and adapt tools natively into LangChain"""
      try:
           if "url" in server_config:
            # Remote server reached over HTTP/SSE
               transport = await self.exit_stack.enter_async_context(
                   sse_client(server_config["url"],timeout=server_config.get("timeout", 5))
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
            for tool in tools:
                #self.tool_to_session[tool.name] = session
                self.sessions[tool.name] = session

            raw_langchain_tools = [
                            convert_mcp_tool_to_langchain_tool(session, tool) for tool in response.tools
                        ]
            for tool in raw_langchain_tools:
             if hasattr(tool, "args_schema") and tool.args_schema:
               schema = tool.args_schema
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
                  tool.args_schema = schema_dict

               tool.description = (tool.description or "").strip()

             self.available_tools.append(tool)

            print("\nConnected to server with tools:", [tool.name for tool in self.available_tools])
                
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
        


    async def connect_to_servers(self):

        """Connect to all configured MCP servers."""
        try:
          with open("server_config.json","r") as file:
            data = json.load(file)
        
          for server_name,server_config in data.get("mcpServers",{}).items():
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
        #create_agent automates parallel calls, self-corrects minor tool exceptions, 
        # and applies protection frameworks against runaway infinite routing loops.
        
        agent = create_agent(
            model = self.llm,
            tools=self.available_tools
        )

        # messages = [{'role':'user','content':'query'}]
        # replaced with LANGCHAIN
        self.messages.append(HumanMessage(content=query))

        t0 = time.time()
        # replaced with LANGCHAIN
        # llm_with_tools = self.llm.bind_tools(self.available_tools)
        # response = await llm_with_tools.ainvoke(self.messages)
        agent_state = await agent.ainvoke({"messages": self.messages})
        print(f"[timing] agent response took {time.time() - t0:.2f}s")

        self.messages = agent_state["messages"]

        # self.messages.append(response)

        # if response.content:
        #     print(f"AI: {response.content}")

        final_response = self.messages[-1]
        if final_response.content:
            print(f"AI: {final_response.content}")

            

                   


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
                print(f"\nError encountered during chat_loop : {str(e)}")

    

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


                
async def main():
    chatbot = MCP_ChatBot()
    try:
        t0 = time.time()
        await chatbot.connect_to_servers() 
        print(f"[timing] server connection took {time.time() - t0:.2f}s")
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() 
  

if __name__ == "__main__":
    asyncio.run(main())