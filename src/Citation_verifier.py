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


