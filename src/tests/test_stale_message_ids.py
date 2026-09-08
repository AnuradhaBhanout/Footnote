from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph.helpers import stale_message_ids

def test_drops_tool_traffic_keeps_conversation():
    msgs = [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[
            {"name": "search", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="{}", id="t1", tool_call_id="tc1"),
        AIMessage(content="here is the answer", id="a2"),
        HumanMessage(content="q2", id="h2"),
    ]
    assert sorted(stale_message_ids(msgs)) == ["a1", "t1"]



def test_keeps_only_recent_turns():
    msgs = [HumanMessage(content=f"q{i}", id=f"h{i}") for i in range(8)]
    assert stale_message_ids(msgs, keep_turns=6) == ["h0", "h1"]