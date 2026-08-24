import asyncio
import uuid
from contextlib import asynccontextmanager


from dotenv import load_dotenv,find_dotenv
from fastapi import FastAPI,HTTPException,Depends,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from api.dependencies import get_chatbot
from api.schemas import ChatRequest,ResumeRequest
from api.sse import stream_graph_events
from client.mcp_v1_chatBot import MCP_ChatBot
from log_setup import setup_logging


 
load_dotenv(find_dotenv())
logger = setup_logging("RAG-API","api-debug.log")





@asynccontextmanager
async def lifespan(app: FastAPI):
    from client.mcp_v1_chatBot import MCP_ChatBot

    chatbot = MCP_ChatBot()
    session_task = asyncio.create_task(chatbot.session_manager())
    app.state.chatbot = chatbot
    app.state.session_task = session_task

    logger.info("API startup complete (connecting in background)")

    try:
        yield
    finally:
        session_task.cancel()
        try:
            await session_task
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


_sse_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}




@app.post("/chat")
async def chat(request: ChatRequest ,chatbot: MCP_ChatBot = Depends(get_chatbot)):  
    "streaming through Langgraph"

    
    session_id = request.session_id or str(uuid.uuid4())
   
    graph_input ={
                "original_query": request.query,
                "current_query": request.query,
                "messages": [],
                "retry_count": 0,
    }


    return StreamingResponse(
        stream_graph_events(chatbot,graph_input,session_id,tags=["chat"]),
        media_type="text/event-stream",     #->>>> Parse and show each line the millisecond it arrives.
        headers=_sse_HEADERS,   
    )



@app.post("/resume")
async def resume(request: ResumeRequest,chatbot:MCP_ChatBot = Depends(get_chatbot)):

   
    return StreamingResponse(
        stream_graph_events(chatbot,Command(resume=request.answer),request.session_id,tags=["resume"]),
        media_type="text/event-stream",
        headers=_sse_HEADERS,
    )




@app.api_route("/health",methods=["GET","HEAD"])
async def health(request: Request):
    chatbot = getattr(request.app.state,"chatbot",None)
    db_ok = False
    if chatbot and hasattr(chatbot, "_pg_pool"):
        try:
            async with chatbot._pg_pool.connection() as conn:
                await conn.execute("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False
    return {"status": "ok", "ready": chatbot is not None and chatbot.ready_event.is_set(), "db": db_ok}