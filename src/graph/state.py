




from typing import TypedDict, Optional,Any




class GraphState(TypedDict):
    original_query: str
    current_query: str
    messages: list
    cache_hit: bool
    draft_answer: Optional[str]
    citation_check_passed: bool
    citation_issues: list
    retry_count: int
    clarification_question: Optional[str]
    clarification_options: list
    answer_is_reliable: bool
    fetched_papers: list 