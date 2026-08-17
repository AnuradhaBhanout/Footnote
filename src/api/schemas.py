from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ResumeRequest(BaseModel):
    session_id: str
    answer: str 