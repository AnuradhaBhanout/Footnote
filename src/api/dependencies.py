"""Returns app's running chatbot, or fail the request with 503"""


from fastapi import HTTPException, Request

from client.mcp_v1_chatBot import MCP_ChatBot

def get_chatbot(request: Request)-> MCP_ChatBot:

    chatbot = getattr(request.app.state, "chatbot", None)
    if not chatbot or not chatbot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Service is not ready")
    return chatbot