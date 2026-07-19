import json
from typing import Any, Optional
 
 
def parse_mcp_content(content: Any) -> Optional[Any]:

    if content is None:
        return None
 
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    if isinstance(content, list):
        texts = [
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        ]
        if not texts:
            return None
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return None
        return texts  # multiple scalar blocks == the original list itself

    return None