import json
from langchain.messages import ToolMessage,HumanMessage
from graph.routing import after_cache,after_citation_check,MAX_RETRIES,after_run_agent,should_continue_searching



BAD = {"results": [], "evaluator_verdict": {"sufficient": False}}
GOOD = {"results": [{"paper_id": "p1"}], "evaluator_verdict": {"sufficient": True}}

def _state(**overrides):
    base =  {
        "original_query": "q",
        "current_query": "q",
        "messages": [],
        "cache_hit": False,
        "draft_answer": None,
        "citation_check_passed": False,
        "citation_issues": [],
        "search_retries": 0,
        "citation_retries":0,
        "clarify_count":0,
        "clarification_question": None,
        "clarification_options": [],
        "answer_is_reliable": False,
        "fetched_papers": [],
    }
    return{**base,**overrides}


def test_after_cache_hit_ends():
    assert after_cache(_state(cache_hit=True)) == "end"

def test_after_cache_miss_continues():
    assert after_cache(_state(cache_hit=False)) == "triage_query"


def test_citation_retries_below_cap_retries():
    assert after_citation_check(_state(citation_retries=MAX_RETRIES-1)) == "retry_with_feedback"

def test_citation_retries_at_cap_falls_back():
    assert after_citation_check(_state(citation_retries=MAX_RETRIES)) == "fallback"

def test_after_citation_check_passed():
    assert after_citation_check(_state(citation_check_passed=True)) == "end"


def test_after_run_agent_no_clarification_needed():
    assert after_run_agent(_state(clarify_count=0)) == "ok"



def test_after_run_agent_first_clarification_allowed():
    state = _state(clarification_question="which temperature?", clarify_count=0)
    assert after_run_agent(state) == "clarify"

def test_after_run_agent_second_clarification_is_blocked():
    state = _state(clarification_question="which temperature?", clarify_count=1)
    assert after_run_agent(state) == "ok"

def _tool_msg(payload: dict):
    return ToolMessage(content=json.dumps(payload), tool_call_id="x")




def test_insufficient_below_cap_retries():
    state = _state(messages=[HumanMessage(content="q"), _tool_msg(BAD)],
                   search_retries=MAX_RETRIES - 1)
    assert should_continue_searching(state) == "retry"

def test_insufficient_at_cap_stops():
    state = _state(messages=[HumanMessage(content="q"), _tool_msg(BAD)],
                   search_retries=MAX_RETRIES)
    assert should_continue_searching(state) == "ok"

def test_empty_results_retries():
    state = _state(messages=[_tool_msg({"results": [], "evaluator_verdict": {"sufficient": True}})],
                   search_retries=0)
    assert should_continue_searching(state) == "retry"

def test_good_results_continue():
    state = _state(messages=[_tool_msg(GOOD)], search_retries=0)
    assert should_continue_searching(state) == "ok"