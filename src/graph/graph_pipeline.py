from typing import TypedDict, Optional,Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage,ToolMessage,trim_messages

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import APIError
import anyio
from mcp.shared.exceptions import McpError
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from client.mcp_content import parse_mcp_content

import httpx
import os
#from structured_outputs import TriageAssessment #QueryReformulation
from db.citation_verifier import verify_citations
import json
import logging
os.makedirs("logs", exist_ok=True)
# Configure logging to write to debug.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/debug.log"),
        logging.StreamHandler() # This also prints to your terminal
    ]
)
logger = logging.getLogger("RAG-Chatbot")

MAX_RETRIES = 2

ACTION_TOOL_NAMES = {"write_file","edit_file","create_directory","move_file","fetch"}



from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse = get_client()   


def _collect_paper_ids_from_search(messages: list) -> list[str]:
    """Pull every paper_id out of search_papers / hybrid_search_papers tool results,
    deduped, in order. Used to drive a single deterministic extract_info call —
    never left to the model to decide how many times to call it."""
    ids = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        data = parse_mcp_content(msg.content)
        # if data is None:
        #     continue
        if not  isinstance(data, dict):
           continue 

        if "results" in data:      # hybrid_search_papers
            verdict = data.get("evaluator_verdict") or {}
            if verdict.get("sufficient") is not True:
                continue
            #ids.extend(r["paper_id"] for r in data.get("results", []) if isinstance(r, dict) and "paper_id" in r)
            SCORE_FLOOR = 0.7
            ids.extend(r["paper_id"] for r in data["results"] 
                       if isinstance(r, dict) and "paper_id" in r and r.get("score",0) >= SCORE_FLOOR)


        elif "paper_ids" in data:
            if not data.get("sufficient"):
                continue
            ids.extend(pid for pid in data["paper_ids"] if isinstance(pid,str))

    seen, out = set(), []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out



def _parse_tool_result(result) -> dict:
    """
    Direct tool.ainvoke() calls (outside the agent's own tool-calling loop) can come back as either
    an already-parsed dict,or as MCP's raw content-block list depending on the adapter.
    Normalize to plain dict either way.
    """
    if isinstance(result,dict):
        return result
    if isinstance(result,list) and result:
        first = result[0]
        text = first.get("text") if isinstance(first,dict) else getattr(first,"text",None)

        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return{}
            
    return {}



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


@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
)
async def _invoke_with_retry(llm, messages):
    return await llm.ainvoke(messages)


def build_graph(llm, chatbot):#, cache_check_tool, cache_store_tool):


    async def check_cache(state: GraphState,config: RunnableConfig)-> GraphState:
        
        await chatbot.acquire_agent()
        try:
            cache_check_tool = next((t for t in chatbot.available_tools if t.name == "check_semantic_cache"), None)
        # 1. Check if the tool actually exists
            if cache_check_tool is None:
                logger.warning("Cache tool missing, skipping cache check.")
                return {**state, "cache_hit": False}

            
            raw = await cache_check_tool.ainvoke({"query": state["original_query"]},config=config)
            result = _parse_tool_result(raw)
            hit = bool(result.get("hit"))
            get_client().score_current_trace(name="cache_hit", value=1 if hit else 0)
            logger.info(f"--- CACHE DEBUG: raw={repr(raw)[:300]}, parsed result={result} ---")
            if hit:
                return {**state, 
                         "cache_hit": True,
                         "draft_answer": result["answer"],
                         "fetched_papers": result.get("fetched_papers", []),
                         "citation_check_passed": True,
                         }
        except Exception as e:
            get_client().score_current_trace(name="error", value=1)       #if e else 0)
            logger.error(f"Cache check failed:{type(e).__name__}: {e}")

        finally:
            await chatbot.release_agent()     
       
        return {**state, "cache_hit": False}
    

    def after_cache(state: GraphState)-> str:
        return "end" if state["cache_hit"] else "triage_query"
    


    

    async def clarify(state: GraphState)-> GraphState:
        """ONly this re-runs on resume - no LLM call here. """
        logger.info("--- NODE START: clarify ---")
        ai_question = state["clarification_question"]

        human_answer= interrupt({
            "question": ai_question,
            "options": state["clarification_options"],
        })

        logger.info(f"--- CLARIFy: RESUMED with: {human_answer}")

        updated_messages = list(state["messages"]) + [
        AIMessage(content=ai_question),
        HumanMessage(content=human_answer),
        ]

        return {
            **state, 
            "messages": updated_messages,
            "current_query": human_answer,
            "clarification_question": None, 
            "clarification_options": [], 
            "retry_count": 0
            }


    async def run_agent(state: GraphState ,config: RunnableConfig) -> GraphState:
        logger.info(f"--- NODE START: run_agent with query: {state['current_query']} ---")
        state = {**state, "fetched_papers": state.get("fetched_papers", [])}
        try:
            messages = state["messages"]

            if not messages or messages[-1].content != state["current_query"]:
                messages = messages + [HumanMessage(content=state["current_query"])]

            messages = trim_messages(
                messages,
                max_tokens=60,
                token_counter=len,
                strategy="last",
                include_system=False,
            )

            async def call_agent(recursion_limit):
                agent = await chatbot.acquire_agent()
                try:
                    merged_config={**config, "recursion_limit": recursion_limit}
                    return await agent.ainvoke({"messages": messages}, config=merged_config)
                    # return await agent.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})
                finally:
                    await chatbot.release_agent()

            try:
                agent_state = await call_agent(8)  #20

            except GraphRecursionError:
                logger.error("run_agent: hit internal recursion limit — agent looped without converging")
                return {
                    **state,
                    "draft_answer": "I couldn't find anything matching that after several attempts — could you try a different phrasing or a known paper title?",
                    "retry_count": state["retry_count"] + 1,
                    "answer_is_reliable": False,
                    "fetched_papers": [],
                }

            except (anyio.ClosedResourceError, McpError):
                for attempt in range(2):
                    chatbot.reconnect_event.set()
                    await chatbot.ready_event.wait()
                    try:
                        agent_state = await call_agent(10)   #22
                        break
                    except (anyio.ClosedResourceError, McpError) as e:
                        if attempt == 1:
                            logger.error(f"run_agent: reconnect retries exhausted: {e}")
                            return {
                                **state,
                                "draft_answer": "The research service is temporarily unavailable — please try again in a moment.",
                                "retry_count": state["retry_count"] + 1,
                                "answer_is_reliable": False, 
                            }
            except httpx.ReadTimeout:
                logger.warning("run_agent: NVIDIA stream stalled, retrying once")
                try:
                    agent_state = await call_agent(10)  #25
                except Exception as e2:
                    logger.error(f"run_agent: retry after timeout also failed: {type(e2).__name__}: {e2}")
                    return {**state,
                            "draft_answer": "The model took too long to respond — please try again.",
                            "retry_count": state["retry_count"] + 1,
                            "answer_is_reliable": False}
                
            except APIError as e:
                if "tool call validation failed" in str(e) or "Failed to call a function" in str(e):
                    logger.warning(f"Malformed tool call, retrying once: {e}")
                    try:
                        agent_state = await call_agent(15)  #55
                    except Exception as e2:
                        logger.error(f"run_agent: retry also failed: {type(e2).__name__}: {e2}")
                        return {**state,
                                "draft_answer": "I had trouble processing that — could you try rephrasing?", 
                                "retry_count": state["retry_count"] + 1,
                                "answer_is_reliable": False}
                else:
                    raise

            except Exception as e:
                # catch-all: anything not matched above skipped straight to
                # agent_state["messages"] and crashed with UnboundLocalError
                logger.error(f"run_agent: unhandled exception from call_agent: {type(e).__name__}: {e}", exc_info=True)
                return {**state,
                        "draft_answer": "I ran into an unexpected issue processing that — please try again.",
                        "retry_count": state["retry_count"] + 1,
                        "answer_is_reliable": False}

            
           
                
    #optimization ( check if the agent asked for clarification instead of searching )
            agent_messages = agent_state["messages"]
            logger.info(f"--- AGENT ACTIONS: {[m.tool_calls for m in agent_state['messages'] if hasattr(m, 'tool_calls') and m.tool_calls]} ---")

            last_ai = next((m for m in reversed(agent_messages) if isinstance(m,AIMessage)and getattr(m,"tool_calls",None)),None)

            if last_ai:
                clarify_call = next((tc for tc in last_ai.tool_calls if tc["name"] == "ask_clarification"),None)
                if clarify_call:
                    args = clarify_call.get("args",{})
                    logger.info(f"----RUN_AGENT: Agent requested clarification: {args.get('question')}")
                    return{
                        **state,
                        "messages": agent_messages,
                        "clarification_question":args.get("question","Could you clarify?"),
                        "clarification_options":args.get("options",[]),
                    }


# --- deterministic extract_info: never left to the model ---
            paper_ids = _collect_paper_ids_from_search(agent_messages)
            logger.info(f"--- DEBUG: collected paper_ids={paper_ids}, raw tool messages={[(type(m.content).__name__, repr(m.content)[:200]) for m in agent_messages if isinstance(m, ToolMessage)]} ---")
            extract_ran = False
            if paper_ids:
                extract_tool = next((t for t in chatbot.available_tools if t.name == "extract_info"), None)
                if extract_tool is not None:
                    try:
                        raw = await extract_tool.ainvoke({"paper_ids": paper_ids},config = config)
                        result = _parse_tool_result(raw)
                        fetched_papers = result.get("papers", []) if isinstance(result, dict) else []
                        extract_msg = ToolMessage(
                            content=json.dumps(result),
                            tool_call_id="deterministic-extract-info",
                            name="extract_info",
                        )
                        agent_messages = agent_messages + [extract_msg]

                        final_pass =[
                            HumanMessage(content=state["current_query"]),
                            HumanMessage(content=f"Paper details:\n{json.dumps(result)}"),
                            HumanMessage(
                            content="Using ONLY the paper details returned above, write your final "
                                    "plain-language summary now. Do not call any tools."
                        ),]
                        final_response = await chatbot.llm.ainvoke(final_pass,config= config)

                        agent_messages = agent_messages + [final_response]
                        extract_ran = True
                    except Exception as e:
                        logger.error(f"run_agent: deterministic extract_info failed: {type(e).__name__}: {e}")
                        # fall through — agent_messages keeps whatever the search-only agent already wrote
            if not extract_ran:
                # covers both "nothing found" and "extract_info call failed" —
                # never let the search-agent's "gathering details" placeholder
                # reach the user as a final answer
                agent_messages = agent_messages + [AIMessage(content=(
                    "I searched but couldn't find papers matching that in your saved library. "
                    "Could you try different terms, or ask me to search again?"
                ))]

            final = agent_messages[-1]
            draft = final.content if final.content else ""
            # final = agent_state["messages"][-1]
            #draft = final.content if final.content else ""
            logger.info("--- NODE END: run_agent completed ---")

            #new_retry_count = state["retry_count"]
            # if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            #     if "[]" in state["messages"][-1].content: # Simple empty check
            #        new_retry_count += 1

            new_retry_count = state["retry_count"]
            last_tool = next(
                (m for m in reversed(agent_state["messages"]) if isinstance(m, ToolMessage)),
                None
            )
            if last_tool:
                #try:
                    # content = last_tool.content
                    # if isinstance(content,list):
                    #     content = next( (b["text"] for b in content if isinstance(b,dict) and b.get("type") == "text"),"")
                        
                    # data = json.loads(content)
                data = parse_mcp_content(last_tool.content)
                if data is not None:
                    results = data.get("results", data.get("papers", [])) if isinstance(data, dict) else data
                    if len(results) == 0:
                        new_retry_count += 1
                # except (json.JSONDecodeError, AttributeError):
                #     pass    
            
            if not draft:
                logger.warning("--- RUN_AGENT: LLM returned empty content ---")
            return {
                **state,
                "messages":agent_messages,  #agent_state["messages"],
                "draft_answer": draft,
                "clarification_question": None,
                "clarification_options": [],
                "retry_count": new_retry_count,
                "answer_is_reliable": extract_ran, #bool(draft),
                "fetched_papers": fetched_papers if extract_ran else [],
                    }
        
        except Exception as e:
            logger.error(f"run_agent: unhandled exception: {type(e).__name__}: {e}", exc_info=True)
            return {
                **state,
                "draft_answer": "I ran into an unexpected issue processing that — please try again.",
                "retry_count": state["retry_count"] + 1,
                "answer_is_reliable": False,
            }
    


    def should_continue_searching(state: GraphState) -> str:
        """
        Conditional edge logic: decide if we need to search again 
        or move to citation checking.
        """
        messages = state.get("messages", [])
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
                
                if is_bad and state["retry_count"] < MAX_RETRIES:
                    logger.info(f"--- LOOP: Search results insufficient. Retrying agent. ---")
                    return "retry"
            # except:
            #     pass # Not JSON or unexpected format, move to citation check

        return "ok"
    

    def after_run_agent(state: GraphState)-> str:   
        if state.get("clarification_question"):
            return "clarify"
        return should_continue_searching(state)


    def check_citations(state: GraphState)-> GraphState:
        result = verify_citations(state["draft_answer"],state["messages"],overlap_threshold=0.3)
        logger.info(f"--- CITATION CHECK: passed={result['passed']} issues={result['issues']}")
        get_client().score_current_trace(name="citation_pass_rate", value=1 if result["passed"] else 0, comment="; ".join(result["issues"]))
        return {**state,"citation_check_passed":result["passed"], "citation_issues":result["issues"]}
    

    
    def _used_actions_tools(state:GraphState)->bool:
        for msg in state["messages"]:
            for tc in getattr(msg,"tool_calls",None) or []:
                if tc["name"] in ACTION_TOOL_NAMES:
                    return True
        return False
    
    def after_citation_check(state: GraphState)-> str:
        if state["citation_check_passed"]:
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
            "answer_is_reliable": False,

        }
    

    async def finalize(state: GraphState,config: RunnableConfig)-> GraphState:
        if state.get("draft_answer") and state["retry_count"] == 0  and state.get("answer_is_reliable", False):
            await chatbot.acquire_agent()
            try:
                cache_store_tool = next((t for t in chatbot.available_tools if t.name == "store_semantic_cache"), None)
        
                if cache_store_tool is not None and state.get("draft_answer")and state["retry_count"] == 0:
                        
                        await cache_store_tool.ainvoke(
                            {
                                "query":state["original_query"],
                                "answer":state["draft_answer"],
                                "fetched_papers": state.get("fetched_papers", []),
                            },
                            config=config
                        )
            except Exception as e:
                logger.error(f"Cache store failed (non-fatal): {e}")
            
            finally:
                await chatbot.release_agent()
        return state
    

    graph = StateGraph(GraphState)
    graph.add_node("check_cache",check_cache)
    #graph.add_node("triage_query",triage_query)
    graph.add_node("run_agent",run_agent)
    #graph.add_node("assess_query", assess_query)
    graph.add_node("clarify", clarify)
    graph.add_node("check_citations",check_citations)
    graph.add_node("retry_with_feedback",retry_with_feedback)
    graph.add_node("fallback",fallback)
    graph.add_node("finalize",finalize)

    graph.set_entry_point("check_cache")
    graph.add_conditional_edges("check_cache",after_cache,{
        "end":END,
        "triage_query": "run_agent"
    })

    # graph.add_conditional_edges("assess_query", after_assess, {
    #     "clarify": "clarify",
    #     "run_agent": "run_agent",
    # })

    #optimization
    graph.add_conditional_edges("run_agent",after_run_agent,{
        "clarify":"clarify",
        "retry":"run_agent",
        "ok":"check_citations",
    })

    graph.add_edge("clarify", "run_agent")

    #graph.add_edge("triage_query","run_agent")
    #graph.add_edge("run_agent","check_citations")
    
    graph.add_conditional_edges(
        "check_citations", after_citation_check,{
            "finalize":"finalize",
            "end_no_cache":END,
            "retry_with_feedback":"retry_with_feedback",
            "fallback":"fallback",
        }
    )

    # graph.add_conditional_edges(
    #     "run_agent",
    #     should_continue_searching,
    #     {
    #         "retry": "run_agent",      # Loop back to agent for a better search
    #         "ok": "check_citations"     # Proceed normally
    #     }
    # )

    graph.add_edge("retry_with_feedback","run_agent")
    graph.add_edge("finalize",END)
    graph.add_edge("fallback",END)

    return graph












    