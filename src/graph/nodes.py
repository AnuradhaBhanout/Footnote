import json
import logging
import anyio
import uuid

import httpx
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage,ToolMessage,trim_messages
from langchain_core.runnables import RunnableConfig

from langfuse import get_client
from langgraph.errors import GraphRecursionError
from langgraph.types import interrupt

from mcp.shared.exceptions import McpError
from openai import APIError

from client.mcp_content import parse_mcp_content
from db.citation_verifier import verify_citations

from graph.helpers import _collect_paper_ids_from_search, _parse_tool_result
from graph.state import GraphState


logger = logging.getLogger("RAG-Chatbot")

class GraphNodes:

    def __init__(self,llm,chatbot):
        self.llm  = llm
        self.chatbot = chatbot

    async def check_cache(self,state: GraphState,config: RunnableConfig)-> GraphState:
        
        await self.chatbot.acquire_agent()
        try:
            cache_check_tool = next((t for t in self.chatbot.available_tools if t.name == "check_semantic_cache"), None)
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
            await self.chatbot.release_agent()     
       
        return {**state, "cache_hit": False}




    async def clarify(self,state: GraphState)-> GraphState:
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






    async def run_agent(self, state: GraphState, config: RunnableConfig) -> GraphState:
        logger.info(f"--- NODE START: run_agent with query: {state['current_query']} ---")
        state = {**state, "fetched_papers": state.get("fetched_papers", [])}
        try:
            messages = self._prepare_agent_messages(state)
            agent_state, fallback_state = await self._invoke_agent_with_recovery(messages, state, config)
            if fallback_state is not None:
                return fallback_state
            return await self._build_agent_response(state, agent_state, config)
 
        except Exception as e:
            logger.error(f"run_agent: unhandled exception: {type(e).__name__}: {e}", exc_info=True)
            return {
                **state,
                "draft_answer": "I ran into an unexpected issue processing that — please try again.",
                "retry_count": state["retry_count"] + 1,
                "answer_is_reliable": False,
            }




    #@staticmethod
    def _prepare_agent_messages(self,state: GraphState) -> list:
            messages = state["messages"]

            if not messages or messages[-1].content != state["current_query"]:
                messages = messages + [HumanMessage(content=state["current_query"])]

            messages = trim_messages(
                messages,
                max_tokens=4000,
                token_counter=self.llm,
                strategy="last",
                include_system=False,
            )
            return messages

    async def _invoke_agent_with_recovery(self, messages: list, state: GraphState, config: RunnableConfig):
        """Runs the agent and handles every failure mode it's known to
        hit. Returns `(agent_state, None)` on success, or
        `(None, fallback_state)` when the failure is handled and
        run_agent should return `fallback_state` as-is."""
        async def call_agent(recursion_limit):
            agent = await self.chatbot.acquire_agent()
            try:
                merged_config={**config, "recursion_limit": recursion_limit}
                return await agent.ainvoke({"messages": messages}, config=merged_config)
                # return await agent.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})
            finally:
                await self.chatbot.release_agent()

        try:
            agent_state = await call_agent(8)  #20

        except GraphRecursionError:
            logger.error("run_agent: hit internal recursion limit — agent looped without converging")
            return None,{
                **state,
                "draft_answer": "I couldn't find anything matching that after several attempts — could you try a different phrasing or a known paper title?",
                "retry_count": state["retry_count"] + 1,
                "answer_is_reliable": False,
                "fetched_papers": [],
            }

        except (anyio.ClosedResourceError, McpError):
            for attempt in range(2):
                self.chatbot.reconnect_event.set()
                await self.chatbot.ready_event.wait()
                try:
                    agent_state = await call_agent(10)   #22
                    break
                except (anyio.ClosedResourceError, McpError) as e:
                    if attempt == 1:
                        logger.error(f"run_agent: reconnect retries exhausted: {e}")
                        return None,{
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
                return None,{**state,
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
                    return None,{**state,
                            "draft_answer": "I had trouble processing that — could you try rephrasing?", 
                            "retry_count": state["retry_count"] + 1,
                            "answer_is_reliable": False}
            else:
                raise

        except Exception as e:
            # catch-all: anything not matched above skipped straight to
            # agent_state["messages"] and crashed with UnboundLocalError
            logger.error(f"run_agent: unhandled exception from call_agent: {type(e).__name__}: {e}", exc_info=True)
            return None,{**state,
                    "draft_answer": "I ran into an unexpected issue processing that — please try again.",
                    "retry_count": state["retry_count"] + 1,
                    "answer_is_reliable": False}
        
        return agent_state,None




    async def _build_agent_response(self, state: GraphState, agent_state: dict, config: RunnableConfig) -> GraphState:
        agent_messages = agent_state["messages"]
        logger.info(
            f"--- AGENT ACTIONS: "
            f"{[m.tool_calls for m in agent_state['messages'] if hasattr(m, 'tool_calls') and m.tool_calls]} ---"
        )
 
        clarification = self._extract_clarification_request(state, agent_messages)
        if clarification is not None:
            return clarification
 
        agent_messages, fetched_papers, extract_ran = await self._run_deterministic_extraction(
            state, agent_messages, config
        )
 
        final = agent_messages[-1]
        draft = final.content if final.content else ""
        logger.info("--- NODE END: run_agent completed ---")
 
        new_retry_count = self._next_retry_count(state, agent_state)
 
        if not draft:
            logger.warning("--- RUN_AGENT: LLM returned empty content ---")
 
        return {
            **state,
            "messages": agent_messages,
            "draft_answer": draft,
            "clarification_question": None,
            "clarification_options": [],
            "retry_count": new_retry_count,
            "answer_is_reliable": extract_ran,
            "fetched_papers": fetched_papers if extract_ran else [],
        }
 
    @staticmethod
    def _extract_clarification_request(state: GraphState, agent_messages: list):
        """If the agent's last move was calling `ask_clarification` instead
        of searching, short-circuit straight to the clarify node. Returns
        the state dict to return early with, or None if there's nothing
        to clarify."""
        last_ai = next(
            (m for m in reversed(agent_messages) if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
            None,
        )
        if not last_ai:
            return None
 
        clarify_call = next((tc for tc in last_ai.tool_calls if tc["name"] == "ask_clarification"), None)
        if not clarify_call:
            return None
 
        args = clarify_call.get("args", {})
        logger.info(f"----RUN_AGENT: Agent requested clarification: {args.get('question')}")
        return {
            **state,
            "messages": agent_messages,
            "clarification_question": args.get("question", "Could you clarify?"),
            "clarification_options": args.get("options", []),
        }
 
    async def _run_deterministic_extraction(self, state: GraphState, agent_messages: list, config: RunnableConfig):
        """extract_info is never left to the model to call — if the search
        turned up paper_ids, we call it exactly once, deterministically,
        then force one more LLM pass to write the final answer from the
        extracted details only."""
        paper_ids = _collect_paper_ids_from_search(agent_messages)
        logger.info(
            f"--- DEBUG: collected paper_ids={paper_ids}, raw tool messages="
            f"{[(type(m.content).__name__, repr(m.content)[:200]) for m in agent_messages if isinstance(m, ToolMessage)]} ---"
        )
 
        fetched_papers = []
        extract_ran = False
        if paper_ids:
            extract_tool = next((t for t in self.chatbot.available_tools if t.name == "extract_info"), None)
            if extract_tool is not None:
                try:
                    raw = await extract_tool.ainvoke({"paper_ids": paper_ids}, config=config)
                    result = _parse_tool_result(raw)
                    fetched_papers = result.get("papers", []) if isinstance(result, dict) else []
                    extract_call_id = f"deterministic-extract-info-{uuid.uuid4().hex[:8]}"
                    extract_ai_msg = AIMessage(
                        content="",
                        tool_calls = [{
                            "name": "extract_info",
                            "args":{"paper_ids": paper_ids},
                            "id": extract_call_id,
                        }],
                    )
                    extract_msg = ToolMessage(
                        content=json.dumps(result),
                        tool_call_id= extract_call_id,
                        name="extract_info",
                    )
                    agent_messages = agent_messages + [extract_ai_msg,extract_msg]
 
                    final_pass = [
                        HumanMessage(content=state["current_query"]),
                        HumanMessage(content=f"Paper details:\n{json.dumps(result)}"),
                        HumanMessage(
                            content="Using ONLY the paper details returned above, write your final "
                            "plain-language summary now. Do not call any tools."
                        ),
                    ]
                    final_response = await self.chatbot.llm.ainvoke(final_pass, config=config)
 
                    agent_messages = agent_messages + [final_response]
                    extract_ran = True
                except Exception as e:
                    logger.error(f"run_agent: deterministic extract_info failed: {type(e).__name__}: {e}")
                    # fall through — agent_messages keeps whatever the search-only agent already wrote
 
        if not extract_ran:
            # covers both "nothing found" and "extract_info call failed" —
            # never let the search-agent's "gathering details" placeholder
            # reach the user as a final answer
            agent_messages = agent_messages + [self._no_results_fallback()]
 
        return agent_messages, fetched_papers, extract_ran
 
    @staticmethod
    def _no_results_fallback() -> AIMessage:
        return AIMessage(content=(
            "I searched but couldn't find papers matching that in your saved library. "
            "Could you try different terms, or ask me to search again?"
        ))
 
    @staticmethod
    def _next_retry_count(state: GraphState, agent_state: dict) -> int:
        new_retry_count = state["retry_count"]
        last_tool = next(
            (m for m in reversed(agent_state["messages"]) if isinstance(m, ToolMessage)),
            None,
        )
        if last_tool:
            data = parse_mcp_content(last_tool.content)
            if data is not None:
                results = data.get("results", data.get("papers", [])) if isinstance(data, dict) else data
                if len(results) == 0:
                    new_retry_count += 1
        return new_retry_count
















    def check_citations(self,state: GraphState)-> GraphState:
        result = verify_citations(state["draft_answer"],state["messages"],overlap_threshold=0.3)
        logger.info(f"--- CITATION CHECK: passed={result['passed']} issues={result['issues']}")
        get_client().score_current_trace(name="citation_pass_rate", value=1 if result["passed"] else 0, comment="; ".join(result["issues"]))
        return {**state,"citation_check_passed":result["passed"], "citation_issues":result["issues"]}


    async def retry_with_feedback(self,state: GraphState) -> GraphState:
        issues_text = "; ".join(state["citation_issues"])
        corrective_query = (
            f"{state['current_query']}\n\n"
            f"IMPORTANT: Your previous answer had citation problem: {issues_text}."
            f"Use ONLY the exact paper_id/title pairs returned by the tools - do not"
            f"Invent or alter any title, author, or finding."
        )
        return {**state,"current_query": corrective_query,"retry_count": state["retry_count"]+1}
    

    def fallback(self,state: GraphState)->GraphState:
        return{
            **state,
            "draft_answer": "I don't have enough verified information to answer that accurately"
            "from your saved papers. Could you rephrase, or ask me to search again wit different terms?",
            "answer_is_reliable": False,

        }
    

    async def finalize(self,state: GraphState,config: RunnableConfig)-> GraphState:

        return state