import asyncio
import uuid
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv,find_dotenv
from fastapi import FastAPI,HTTPException,Depends,Request,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from langfuse import get_client

from api.dependencies import get_chatbot
from api.schemas import ChatRequest,ResumeRequest,FeedbackRequest
from api.sse import stream_graph_events
from client.mcp_v1_chatBot import MCP_ChatBot
from log_setup import setup_logging
from db.db import init_db
from server.mcp_app import mcp
import server.tools

from slowapi import Limiter,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from db.db import get_pool


 
load_dotenv(find_dotenv())
logger = setup_logging("RAG-API","api-debug.log")





@asynccontextmanager
async def lifespan(app: FastAPI):

    chatbot = MCP_ChatBot()
    init_db()                            # creates tables
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


limiter = Limiter(key_func= get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

# a security filter that intercepts incoming requests before they reach your endpoints
ALLOWED_ORIGINS=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],

)

app.mount("/mcp", mcp.sse_app())


_sse_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}




@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest, chatbot: MCP_ChatBot = Depends(get_chatbot)):  
    "streaming through Langgraph"

    
    session_id = str(body.session_id) if body.session_id else str(uuid.uuid4())
   
    graph_input ={
                "original_query": body.query,
                "current_query": body.query,
                "messages": [],
                "search_retries":0,
                "citation_retries":0,
                "clarify_count":0,
    }


    return StreamingResponse(
        stream_graph_events(chatbot,graph_input,session_id,tags=["chat"]),
        media_type="text/event-stream",     #->>>> Parse and show each line the millisecond it arrives.
        headers=_sse_HEADERS,   
    )




@app.post("/resume")
@limiter.limit("10/minute")
async def resume(request:Request, body: ResumeRequest,chatbot:MCP_ChatBot = Depends(get_chatbot)):
    session_id = str(body.session_id)
    state = await chatbot.app.aget_state({"configurable": {"thread_id":session_id}})
    if not state.next:
        raise HTTPException(404, "No paused session with that id")
   
    return StreamingResponse(
        stream_graph_events(chatbot,Command(resume=body.answer),session_id,tags=["resume"]),
        media_type="text/event-stream",
        headers=_sse_HEADERS,
    )




@app.api_route("/health",methods=["GET","HEAD"])
async def health(request: Request, response: Response):
    chatbot = getattr(request.app.state,"chatbot",None)
    db_ok = False
    try:
        conn = get_pool().getconn()
        try:
            with conn.cursor() as cur:
              cur.execute("SELECT 1")
            db_ok = True
        finally:
          get_pool().putconn(conn)
    except Exception:
        db_ok = False

    if not db_ok:
        response.status_code = 503
    return {"status": "ok" if db_ok else "degraded", "ready": chatbot is not None and chatbot.ready_event.is_set(), "db": db_ok}




@app.post("/feedback")
@limiter.limit("10/minute")
async def feedback(request:Request, body: FeedbackRequest):
    try:
     get_client().create_score(trace_id=body.trace_id,
                              name="user_feedback",
                              value=1 if body.is_positive else 0)

    except Exception as e:
        logger.warning("feedback score failed",exc_info=True)
    return {"ok":True}