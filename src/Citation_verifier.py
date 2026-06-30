import re
import json
import sys
import logging
from langchain_core.messages import ToolMessage

logger = logging.getLogger("RAG-Chatbot")

ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)\b", re.IGNORECASE)


def extract_real_papers_from_tool_results(messages: list) -> dict:
    real_papers = {}

    # Step 1: from AIMessage tool_calls, map tool_call_id -> paper_id for extract_info calls
    extract_info_map = {}  # {tool_call_id: paper_id}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "extract_info":
                extract_info_map[tc["id"]] = tc["args"].get("paper_id", "")

    # Step 2: match ToolMessages back to paper_ids using tool_call_id, extract title from JSON
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        paper_id = extract_info_map.get(msg.tool_call_id)
        if not paper_id:
            continue
        try:
            content = msg.content
            if isinstance(content, list):
                content = next((b["text"] for b in content if isinstance(b,dict) and b.get("type") == "text"),"")
                
            data = json.loads(content)
            title = data.get("title","")
            if title:
                real_papers[paper_id] = title
        except (json.JSONDecodeError, AttributeError,StopIteration):
            pass

    # Step 3: also handle search_papers results which may include paper_id fields directly
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            content = msg.content
            if isinstance(content, list):
                content = next((b["text"] for b in content if isinstance(b,dict) and b.get("type") == "text"),"")
            data = json.loads(content)
            papers = data.get("results", data.get("papers", []))
            for p in papers:
                if isinstance(p, dict) and "paper_id" in p and "title" in p:
                    real_papers[p["paper_id"]] = p["title"]
        except (json.JSONDecodeError, AttributeError,StopIteration):
            pass

    logger.info(f"--- CITATION VERIFIER: real_papers extracted = {list(real_papers.keys())}")
    return real_papers

def _title_overlap_ration(real_title: str,answer_text: str) -> float:
    """
    What graction of the real title's significant words actually 
    appear in the answer.
    """
    significant_words = [w.lower().strip(".,:;\"'()")  for w in real_title.split() if len(w)>3 ]
    if not significant_words:
        return 1.0
    
    answer_lower = answer_text.lower()
    matched = sum(1 for w in significant_words if w in answer_lower)

    return matched/len(significant_words)


def verify_citations(answer_text:str,messages:list,overlap_threshold: float = 0.4) -> dict:
    """
    Returns {"passed": bool, "issues": [str, ...]}.
    Fails if: a cited paper_id never appeared in any real tool result(fabricated ID),
    or a cited paper_id's real title shares too little overlap with the answer text
    (right ID, wrong/invented title or findings - the harder fabrication case).
    """
    real_papers = extract_real_papers_from_tool_results(messages)
    cited_ids = set(ARXIV_ID_PATTERN.findall(answer_text))

    if not cited_ids:
        return {"passed":True,"issues": []}
    
    issues = []
    for paper_id in cited_ids:
        if paper_id not in real_papers:
            issues.append(f"Cited paper_id '{paper_id}' was never returned by any tool - likely fabricated.")
            continue

        real_title= real_papers[paper_id]
        overlap = _title_overlap_ration(real_title,answer_text)
        if overlap < overlap_threshold:
            issues.append(f"paper '{paper_id}' is real, but its actual title (\"{real_title}\")")
            f"barely appears in the answer (overlap={overlap: .2f}) - title/findings may be invented."

    return{"passed": len(issues) == 0, "issues":issues}