from pydantic import BaseModel,Field


class ChatRequest(BaseModel):
    query: str =Field(min_length=1,max_length=2000)
    session_id: str | None = Field(default=None,max_length=64)

class ResumeRequest(BaseModel):
    session_id: str=Field(max_length=64)
    answer: str = Field(min_length=1,max_length=500)