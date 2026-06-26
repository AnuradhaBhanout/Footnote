from typing import TypedDict, Optional,Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage

from structured_outputs import TriageAssessment, QueryReformulation
from Citation_verifier import verify_citation

MAX_RETRIES = 2

ACTION_TOOL_NAMES = {"write_file","edit_file","create_directory","move_file","fetch"}

TRIAGE_SYSTEM_PROMPT = """
You are a triage assistant for a research helper tool used by everyday\people, not 
just technical experts. Your only job: decide if the user;s request is specific\
enough to act on right away, or if it's ambigious and needs a quick clarifying question
first.

A request is UNCLEAR if any of these are true:
- It uses a short term,name,or abbrivation that could plausibly mean more than one real thing.
- It refers to "the paper," "that one," "this," or similar without saying which one, when more
than one thing could be meant.
- It's too broad or vauge to search well(a single very general word covering many unrelated sub-topics).
- It's missing a detail that would clearly change the answer or action (e.g. asking to save something without saying what or where).

A request is CLEAR if it names a specific-enough topic or action that a reasonable a person could act on without guessing.

Write any carifiying question in plain, friendly language - do not assume the user knows technical jargon or this system's internal terms.

"""


class GraphState(TypedDict):
    original_query: str
    current_query: str
    messages: list
    cache_hit: bool
    draft_answer: Optional[str]
    citaation_check_passed: bool
    citation_issues: list
    retry_count: int

def build_graph(llm,agent,cache_check_tool, cache_store_tool):
     
    triage_llm = llm.with_structured_output(TriageAssessment)
    reformulate_llm = llm.with_structured_output(QueryReformulation)

    async def check_cache(state: GraphState)-> GraphState:
        result = await cache_check_tool.ainvoke({"query": state["original_query"]})
        if result.get("hit"):
            return{**state, "cache_hit":True,"draft_answer":result["answer"]}
        return {**state,"cache_hit":False}
    
    def after_cache(state: GraphState)-> str:
        return "end" if state["cache_hit"] else "triage_query"
    
    async def triage_query(state: GraphState) -> GraphState:
        assessment: TriageAssessment = await triage_llm.ainvoke([
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=state["original_query"]),
        ])

# Implementing HUMAN IN LOOP
        if not assessment.is_clear:
            human_answer = interrupt({
                "question": assessment.clarifying_question,
                "options": assessment.possible_interpretations,
            })
            return {**state,"current_query":human_answer,"retry_count":0}
        
        return{**state,"current_query": state["original_query"],"retry_count":0}