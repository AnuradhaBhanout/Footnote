import re
import json
import sys
import logging
from langchain_core.messages import ToolMessage
from client.mcp_content import parse_mcp_content

logger = logging.getLogger("RAG-Chatbot")

ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)\b", re.IGNORECASE)

def _strip_version(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id)




def extract_real_papers_from_tool_results(messages: list) -> dict:
    real_papers = {}

    # Step 1: Extract directly from ToolMessage output content.
    # Because extract_info now returns {"papers": [{"paper_id": "...", "title": "..."}]}
    # the tool-call map matching of call_id to paper_id is no longer needed.
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        if getattr(msg, "name", None) != "extract_info":
            continue
        data = parse_mcp_content(msg.content)
            # Handle new plural shape: {"papers": [{"paper_id": "...", "title": "..."}, ...]}
        if isinstance(data, dict) and "papers" in data:
            for p in data["papers"]:
                if isinstance(p, dict) and "paper_id" in p and "title" in p:
                    real_papers[_strip_version(p["paper_id"])] = p["title"]
        # except (json.JSONDecodeError, AttributeError, StopIteration):
        #     pass

    # Step 2: Fallback to old singular extract_info shape matching for backward compatibility
    extract_info_map = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "extract_info":
                # Only map if singular argument was used
                paper_id = tc["args"].get("paper_id", "")
                if paper_id:
                    extract_info_map[tc["id"]] = paper_id

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        paper_id = extract_info_map.get(msg.tool_call_id)
        if not paper_id:
            continue

        data = parse_mcp_content(msg.content)
        if isinstance(data, dict):
            title = data.get("title", "")
            if title:
                real_papers[_strip_version(paper_id)] = title
        # except (json.JSONDecodeError, AttributeError, StopIteration):
        #     pass

    # Step 3: Extract from search_papers & hybrid_search_papers results
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        data = parse_mcp_content(msg.content)
        if isinstance(data, dict):
            papers = data.get("results", data.get("papers", []))
            for p in papers:
                if isinstance(p, dict) and "paper_id" in p and "title" in p:
                    real_papers[_strip_version(p["paper_id"])] = p["title"]
        # except (json.JSONDecodeError, AttributeError, StopIteration):
        #     pass

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
    #cited_ids = set(ARXIV_ID_PATTERN.findall(answer_text))
    cited_ids = {_strip_version(pid) for pid in ARXIV_ID_PATTERN.findall(answer_text)}

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
            issues.append(
            f"paper '{paper_id}' is real, but its actual title (\"{real_title}\")"
            f"barely appears in the answer (overlap={overlap: .2f}) - title/findings may be invented."
            )

    return{"passed": len(issues) == 0, "issues":issues}