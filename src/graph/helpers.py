import json
from langchain_core.messages import ToolMessage

from client.mcp_content import parse_mcp_content

SCORE_FLOOR = 0.7

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