import re

ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)\b",re.IGNORECASE)


def extract_real_papers_from_tool_results(messages: list) -> dict:
    """
    Walk the agents's message history and build a {paper_id: title} map from what tools ACTUALLY
    returned - this is the ground truth the final answer gets checked against, independent of what
    the model claims.
    """
    real_papers = {}

    for msg in messages:
        # pull paper_id/title pairs
        content = getattr(msg,"content",None)
        if not content or not isinstance(content,str):
            continue

        for match in re.finditer(r'"paper_id"\s*:\s*"([^"]+)"[^}]*?"title"\s*([^"]+)"',content):
            real_papers[match.group(1)] = match.group(2)

        for match in re.finditer(r'"title"\s*:\s*"([^"]+)"[^}]*?"paper_id"\s*:\s*"([^"]+)"',content):
            real_papers[match.group(2)] = match.group(1)

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


def verify_citation(answer_text:str,messages:list,overlap_threshold: float = 0.4) -> dict:
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