#the streaming logic

import json
import logging  

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logger = logging.getLogger("RAG-API")
langfuse = get_client

TRUNCATE_EXEMPT_TOOLS = ("hybrid_search_papers","search_papers","extract_info")

def sse_event(event: str, data: dict)-> str:
    return f"event:{event}\ndata:{json.dumps(data)}\n\n"

def format_tool_output(name: str, output) -> str:
    if hasattr("output","content"):
        output = output.content
    if not isinstance(output,str):
       try:
           output = json.dumps(output)
       except(TypeError,ValueError):
           output = str(output)

    if name not in TRUNCATE_EXEMPT_TOOLS and len(output)>300:
        output = output[:300] + "..."
    return output