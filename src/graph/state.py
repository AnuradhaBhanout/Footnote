




from typing import TypedDict, Optional,Any,Annotated
from langgraph.graph.message import add_messages




class GraphState(TypedDict):
    original_query: str
    current_query: str
    messages: Annotated[list, add_messages]
    cache_hit: bool
    draft_answer: Optional[str]
    citation_check_passed: bool
    citation_issues: list
    clarification_question: Optional[str]
    clarification_options: list
    answer_is_reliable: bool
    fetched_papers: list 
    search_retries: int
    citation_retries:int
    clarify_count: int