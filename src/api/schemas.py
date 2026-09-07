from pydantic import BaseModel,Field
import uuid

class ChatRequest(BaseModel):
    query: str =Field(min_length=1,max_length=2000)
    session_id: uuid.UUID | None = None

class ResumeRequest(BaseModel):
    session_id: uuid.UUID=Field(max_length=64)
    answer: str = Field(min_length=1,max_length=500)