import asyncio
import json
import logging
import os
import selectors
import uuid

import psycopg
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from dotenv import load_dotenv,find_dotenv

from contextlib import asynccontextmanager
from citation_verifier import extract_real_papers_from_tool_results, ARXIV_ID_PATTERN

load_dotenv(find_dotenv())

from graph_pipeline import build_graph
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RAG-API")



DATABASE_URL = os.getenv("DATABASE_URL")

_app_state: dict = {}    #holds llm,agent, graph_app, pg_conn, tools



# @asynccontextmanager
# async def lifespan(app: FastAPI):

#     from mcp_v1_chatBot import MCP_ChatBot

#     chatbot = MCP_ChatBot()
#     await chatbot.connect_to_servers()      #connect to mcp server
#     await chatbot._build_agent_and_graph()   #build agecnts + graph once at startup

#     _app_state["chatbot"] = chatbot
    
#     # async def _keepalive():
#     #     while True:
#     #         await asyncio.sleep(30)
#     #         try:
#     #             session = next(iter(chatbot.sessions.values()), None)
#     #             if session:
#     #                 await session.send_ping()
#     #         except Exception:
#     #             pass
#     #asyncio.create_task(_keepalive())
#     task = asyncio.create_task(chatbot.session_manager())
#     await chatbot.ready_event.wait()
#     _app_state["chatbot"] = chatbot
#     _app_state["session_task"] = task

#     logger.info("API startup complete")

#     try:
#         yield  # server is now live and handles requests
        
#     finally:
#         # no matter what if app runs normal or crashed i will shutdown the server.
#         # chatbot = _app_state.get("chatbot")  
#         # if chatbot:
#         #     await chatbot.cleanup()

#         # logger.info("API shutdown complete.")

#         task = _app_state.get("session_task")
#         if task:
#             task.cancel()
#             try:
#                 await task
#             except asyncio.CancelledError:
#                 pass
#         logger.info("API shutdown complete.")



@asynccontextmanager
async def lifespan(app: FastAPI):
    from mcp_v1_chatBot import MCP_ChatBot

    chatbot = MCP_ChatBot()
    task = asyncio.create_task(chatbot.session_manager())
    _app_state["chatbot"] = chatbot
    _app_state["session_task"] = task

    logger.info("API startup complete (connecting in background)")

    try:
        yield
    finally:
        task = _app_state.get("session_task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("API shutdown complete.")

        



#APP
app = FastAPI(title="RAGchatbot API", version="1.0.0",lifespan=lifespan)

# a security filter that intercepts incoming requests before they reach your endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # will tighten in production
    allow_methods=["*"],
    allow_headers=["*"],

)



class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ResumeRequest(BaseModel):
    session_id: str 
    answer: str


def sse_event(event: str,data: dict)-> str:

    return f"event:{event}\ndata:{json.dumps(data)}\n\n"


TRUNCATE_EXEMPT_TOOLS = ("hybrid_search_papers", "search_papers", "extract_info")

def format_tool_output(name: str, output) -> str:
    if hasattr(output, "content"):       # ToolMessage
        output = output.content
    if not isinstance(output, str):
        try:
            output = json.dumps(output)
        except (TypeError,ValueError):
            output = str(output)
        
    if name not in TRUNCATE_EXEMPT_TOOLS and len(output) > 300:
        output = output[:300] + "..."
    return output


@app.post("/chat")
async def chat(request: ChatRequest):
    "streaming through Langgraph"

    chatbot = _app_state.get("chatbot")
    if not chatbot or not chatbot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Service is not ready")
    
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable":{"thread_id": session_id}}   # CHECKPOINTER

    async def event_stream():
        try:
            async for event in chatbot.app.astream_events(
                {
                    "original_query": request.query,
                    "current_query": request.query,
                    "messages": [],
                    "retry_count": 0,
                },
                config,
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name","")

                if kind == "on_tool_start":
                    yield sse_event("tool_start",{
                        "tool": name,
                        "input": event.get("data", {}).get("input",{}),
                    })

                elif kind == "on_tool_end":
                    output = format_tool_output(name, event.get("data",{}).get("output",""))
                    input_args = event.get("data",{}).get("input",{})
                    yield sse_event("tool_end",{
                        "tool":name,
                        "output":output,
                        "input": input_args,
                    })
         
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data",{}).get("chunk")
                    if chunk and hasattr(chunk,"content") and chunk.content:
                        yield sse_event("token",{"content": chunk.content})

                
               

                
            state = await chatbot.app.aget_state(config)
            
            # on graph interrupt (triage)
            if state.next:
                interrupt_data = state.tasks[0].interrupts[0].value
                yield sse_event("interrupt",{
                    "question": interrupt_data.get("question", "Could you please specify?"),
                    "options": interrupt_data.get("options",[]),
                    "session_id": session_id,
                })
                return        # pause
            
            answer = state.values.get("draft_answer","")
            cited_ids = list(
                set(ARXIV_ID_PATTERN.findall(answer)) 
                & set(extract_real_papers_from_tool_results(state.values.get("messages", [])).keys()))

            yield sse_event("done",{
                "answer": answer,
                "session_id": session_id,
                "cited_paper_ids": cited_ids,
            })
        
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            user_message = "Something went wrong processing your request. Please try again."
            yield sse_event("error",{"message":str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",     #->>>> Parse and show each line the millisecond it arrives.
        headers={
            "Cache-Control": "no-cache",   # Hey, CDN "Do not store or save a copy of this connection, open a live connection"
            "X-Accel-Buffering": "no",     # Hey, Nginx  push every single character to the user instantly,do not wait to make a larger block of text .
        },
    )



@app.post("/resume")
async def resume(request: ResumeRequest):

    chatbot = _app_state.get("chatbot")
    if not chatbot:
        raise HTTPException(status_code=503,detail="Service not ready")
    
    config = {"configurable":{"thread_id":request.session_id}}

    async def event_stream():
        try:
            async for event in chatbot.app.astream_events(
                Command(resume=request.answer),
                config,
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name","")

                if kind == "on_tool_start":
                    yield sse_event("tool_start",{
                        "tool":name,
                        "input":event.get("data",{}).get("input",{}),
                    })
                
                elif kind == "on_tool_end":
                    output = format_tool_output(name, event.get("data",{}).get("output",""))
                    input_args = event.get("data",{}).get("input",{})
                    yield sse_event("tool_end",{
                        "tool":name,
                        "output":output,
                        "input": input_args,
                    })

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data",{}).get("chunk")
                    if chunk and hasattr(chunk,"content") and chunk.content:
                        yield sse_event("token",{"content":chunk.content})

                    
            state = await chatbot.app.aget_state(config)

            # on graph interrupt (triage)
            if state.next:
                interrupt_data = state.tasks[0].interrupts[0].value
                yield sse_event("interrupt",{
                    "question": interrupt_data.get("question", "Could you please specify?"),
                    "options": interrupt_data.get("options",[]),
                    "session_id": request.session_id,
                })
                return        # pause
            
            answer = state.values.get("draft_answer","")

            cited_ids = list(
                set(ARXIV_ID_PATTERN.findall(answer)) 
                & set(extract_real_papers_from_tool_results(state.values.get("messages", [])).keys()))
            
            yield sse_event("done",{
                "answer": answer,
                "session_id": request.session_id,
                "cited_paper_ids": cited_ids,
            })
        
        except Exception as e:
            logger.error(f"Resume stream error: {e}",exc_info=True)
            yield sse_event("error",{"message":str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/health")
async def health():
    return {"status":"ok",
            "ready":"chatbot" in _app_state}