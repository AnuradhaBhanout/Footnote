

import json
import logging
import os
import re
from openai import OpenAI



_evaluator_client =OpenAI(
    base_url = "https://api.cerebras.ai/v1",                                 
    api_key= os.getenv("CEREBRAS_API_KEY"),                                 
    timeout=15.0,
)

EVALUATOR_MODEL =  "gpt-oss-120b"   

QUOTED_TITLE_PATTERN = re.compile(r'["\u201c]([^"\u201d]{4,})["\u201d]')

def _quoted_phrase(query: str) -> str | None:
    m = QUOTED_TITLE_PATTERN.search(query)
    return m.group(1) if m else None




def evaluate_relevance(query:str,results:list)->dict:
    """LLM-as-judge: is at least one retrieved paper actually relevant, or is this a bad batch?"""
    if not results:
        return{"sufficient": False,"best_paper_id": None,"reason":"No results retrieved."}
    
    candidates = "\n".join(f"- {r['paper_id']}: {r['title']}\n {r.get('summary','')[:200]}" for r in results)

    prompt = f"""You are a strict relevance judge. Query: "{query}"

    Retrieved papers:{candidates}
    Does AT LEAST ONE paper genuinely answer the query - not just share the few words with it?
    Respond with ONLY this JSON, nothing else:
    {{"sufficient":true or false,"best_paper_id":"<id or null>","reason":"<one sentence>"}}
    """
    response = _evaluator_client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=300,
        reasoning_effort="low",
        response_format={"type":"json_object"},
    )

    try:
        # return json.loads(response.choices[0].message.content)
        #response = _relevance_judge.invoke(prompt)
        content = response.choices[0].message.content.strip()
        
        # --- STRIP MARKDOWN BACKTICKS FOR ROBUST PARSING ---
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        # ---------------------------------------------------
        return json.loads(content)
    
    except (json.JSONDecodeError,AttributeError) as e:
        # Log the raw text to see what the LLM returned on parse failure
        logging.error(f"[evaluator parse error]: {str(e)} - raw text: {response.choices[0].message.content if response.choices else 'No choices available'}")
        return{"sufficient":False,
               "best_paper_id":results[0]["paper_id"]if results else None,
               "reason":"Judge Parse failure - defaulted to top result."}