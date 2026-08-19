"""
Run from src/: uv run test_triage_structured.py
Needs OPENAI_API_KEY (OpenRouter key) in your .env — calls the real model.
"""
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from structured_outputs import TriageAssessment
from graph_pipeline import TRIAGE_SYSTEM_PROMPT

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="liquid/lfm-2.5-1.2b-thinking:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
triage_llm = llm.with_structured_output(TriageAssessment,method="function_calling")


def run(query: str):
    result: TriageAssessment = triage_llm.invoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    print(f"Query: {query!r}")
    print(f"  is_clear={result.is_clear}")
    print(f"  reason={result.reason}")
    if not result.is_clear:
        print(f"  clarifying_question={result.clarifying_question}")
        print(f"  possible_interpretations={result.possible_interpretations}")
    print()


print("--- Test 1: clear query (expect is_clear=True) ---")
run("find papers about transformer attention mechanisms")

print("--- Test 2: ambiguous query (expect is_clear=False) ---")
run("search papers on POPE")
