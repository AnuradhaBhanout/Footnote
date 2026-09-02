import logging
from langchain_core.messages import ToolMessage

from client.mcp_content import parse_mcp_content
from graph.state import GraphState
from graph.helpers import _current_turn_messages

logger = logging.getLogger("RAG-CHATBOT")

MAX_RETRIES = 2



def after_cache(state: GraphState)-> str:
    return "end" if state["cache_hit"] else "triage_query"


def should_continue_searching(state: GraphState) -> str:
    """
    Conditional edge logic: decide if we need to search again 
    or move to citation checking.
    """
    messages = _current_turn_messages(state.get("messages", []))
    if not messages:
        return "ok"

    # Find the last message from a tool
    last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    
    if last_tool_msg:
        
        data = parse_mcp_content(last_tool_msg.content)
        if data is not None:
            
            # Check for 'insufficient' verdict OR completely empty results list
            verdict = data.get("evaluator_verdict", {}) if isinstance(data,dict) else {}
            results = data.get("results", data.get("papers", [])) if isinstance(data, dict) else data

            is_bad = verdict.get("sufficient") is False or len(results) == 0
            
            if is_bad and state["search_retries"] < MAX_RETRIES:
                logger.info(f"--- LOOP: Search results insufficient. Retrying agent. ---")
                return "retry"
        # except:
        #     pass # Not JSON or unexpected format, move to citation check

    return "ok"



def after_run_agent(state: GraphState) -> str:
    if state.get("clarification_question") and state.get("clarify_count", 0) < 1:
        return "clarify"
    return should_continue_searching(state)




def after_citation_check(state: GraphState)-> str:
    if state["citation_check_passed"]:
        return "end"
    if state["citation_retries"] >= MAX_RETRIES:
        return "fallback"
    return "retry_with_feedback"
    