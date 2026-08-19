#the streaming logic

import json
import logging  

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logger = logging.getLogger("RAG-API")
langfuse = get_client()

TRUNCATE_EXEMPT_TOOLS = ("hybrid_search_papers","search_papers","extract_info")

def sse_event(event: str, data: dict)-> str:
    return f"event:{event}\ndata:{json.dumps(data)}\n\n"

def format_tool_output(name: str, output) -> str:
    if hasattr("output","content"):
        output = output.content
    if not isinstance(output,str):
       try:
           output = json.dumps(output)
       except(TypeError,ValueError):
           output = str(output)

    if name not in TRUNCATE_EXEMPT_TOOLS and len(output)>300:
        output = output[:300] + "..."
    return output


async def stream_graph_events(chatbot, graph_input, session_id: str, tags: list[str]):

    handler = CallbackHandler()
    with langfuse.start_as_current_observation(as_type="span",name="chat-request") as span,\
    propagate_attributes(session_id=session_id, user_id=session_id, tags=tags):
        config ={"configurable":{"thread_id":session_id},"callbacks":[handler]}

        try:
            async for event in chatbot.app.astream_events(graph_input,config,version="v2"):
                kind = event["event"]
                name = event.get("name","")

                if kind == "on_tool_start":
                    yield sse_event("tool_start",{
                        "tool":name,
                        "input":event.get("data",{}).get("input",{}),
                    })
                elif kind =="on_tool_end":
                    output = format_tool_output(name,event.get("data",{}).get("output",""))
                    input_args = event.get("data",{}).get("input",{})
                    yield sse_event("tool_end",{
                        "tool":name,
                        "output":output,
                        "input": input_args,
                    })
        
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data",{}).get("chunk")
                    if chunk and hasattr(chunk,"content") and chunk.content:
                        yield sse_event("token",{"content": chunk.content})

            state = await chatbot.app.aget_state(config)

            # on graph interrupt (triage)
            if state.next:
                interrupt_data = state.tasks[0].interrupts[0].value
                yield sse_event("interrupt",{
                    "question": interrupt_data.get("question", "Could you please specify?"),
                    "options": interrupt_data.get("options",[]),
                    "session_id": session_id,
                })
                return        # pause
            
            answer = state.values.get("draft_answer","")

            citation_passed = state.values.get("citation_check_passed", False)
            
            answer_is_reliable = state.values.get("answer_is_reliable",False)
            fetched_papers = state.values.get("fetched_papers", []) if answer_is_reliable else []
            cited_ids = [pid["paper_id"] for pid in fetched_papers if isinstance(pid,dict) and "paper_id" in pid] if citation_passed else []

            trace_id = span.trace_id
            yield sse_event("done",{
                "answer": answer,
                "session_id": session_id,
                "cited_paper_ids": cited_ids,
                "fetched_papers": fetched_papers,
                "trace_id": trace_id,
            })
        
        except Exception as e:
            logger.error(f"stream error: {e}",exc_info=True)
            yield sse_event("error",{"message":str(e)})