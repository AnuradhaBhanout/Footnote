from typing import TypedDict, Optional,Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage

from structured_outputs import TriageAssessment, QueryReformulation
from citation_verifier import verify_citation

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
    


    async def run_agent(state: GraphState)-> GraphState:
        messages = state["messages"]+[HumanMessage(content=state["current_query"])]
        agent_state = await agent.ainvoke({"messages":messages})
        final = agent_state["messages"][-1]
        return {**state,"messages":agent_state["messages"],"draft_answer": final.conent}
    
    def check_citations(state: GraphState)-> GraphState:
        result = verify_citation(state["draft_answer"],state["messages"])
        return {**state,"citation_check_passed":result["passed"], "citation_issues":result["issues"]}
    
    def _used_actions_tools(state:GraphState)->bool:
        for msg in state["messages"]:
            for tc in getattr(msg,"tool_calls",None) or []:
                if tc["name"] in ACTION_TOOL_NAMES:
                    return True
        return False
    
    def after_citation_check(state: GraphState)-> str:
        if state["citaation_check_passed"]:
            return "end_no_cache" if _used_actions_tools(state) else "finalize"
        if state["retry_count"] >= MAX_RETRIES:
            return "fallback"
        return "retry_with_feedback"
    


    async def retry_with_feedback(state: GraphState) -> GraphState:
        issues_text = "; ".join(state["citation_issues"])
        corrective_query = (
            f"{state['current_query']}\n\n"
            f"IMPORTANT: Your previous answer had citation problem: {issues_text}."
            f"Use ONLY the exact paper_id/title pairs returned by the tools - do not"
            f"Invent or alter any title, author, or finding."
        )
        return {**state,"current_query": corrective_query,"retry_count": state["retry_count"]+1}
    

    def fallback(state: GraphState)->GraphState:
        return{
            **state,
            "draft_answer": "I don't have enough verified information to answer that accurately"
            "from your saved papers. Could you rephrase, or ask me to search again wit different terms?",

        }
    

    async def finalize(state: GraphState)-> GraphState:
        await cache_store_tool.ainvoke(
            {
                "query":state["original_query"],
                "answer":state["draft_answer"]
            }
        )
        return state
    

    graph = StateGraph(GraphState)
    graph.add_node("check_cache",check_cache)
    graph.add_node("triage_query",triage_query)
    graph.add_node("run_agent",run_agent)
    graph.add_node("check_citations",check_citations)
    graph.add_node("retry_with_feedback",retry_with_feedback)
    graph.add_node("fallback",fallback)
    graph.add_node("finalize",finalize)

    graph.set_entry_point("check_cache")
    graph.add_conditional_edges("check_cache",after_cache,{
        "end":END,
        "triage_query": "triage_query"
    })
    graph.add_edge("triage_query","run_agent")
    graph.add_edge("run_agent","check_citations")
    graph.add_conditional_edges(
        "check_citations", after_citation_check,{
            "finalize":"finalize",
            "end_no_cache":END,
            "retry_with_feedback":"retry_with_feedback",
            "fallback":"fallback",
        }
    )

    graph.add_edge("retry_with_feedback","run_agent")
    graph.add_edge("finalize",END)
    graph.add_edge("fallback",END)

    return graph












    