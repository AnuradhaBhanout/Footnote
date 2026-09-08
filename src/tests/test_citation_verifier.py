from db.citation_verifier import verify_citations


from langchain_core.messages import ToolMessage
import json

def test_real_citation_is_verified():
    messages = [ToolMessage(
        content=json.dumps({"papers": [
            {"paper_id": "2502.10881v1",
            "title": "CiteCheck: Towards Accurate Citation Faithfulness Detection"}
        ]}),
        name="extract_info",
        tool_call_id="tc1",
    )]
    answer = "The paper 2502.10881 (CiteCheck: Towards Accurate Citation Faithfulness) shows..."
    r = verify_citations(answer, messages)
    assert r["verified"] is True
    assert r["passed"] is True

def test_no_citations_is_not_verified():
    r = verify_citations("plain text with no ids", [])
    assert r["passed"] is True
    assert r["verified"] is False