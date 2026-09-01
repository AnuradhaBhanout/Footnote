"""
Regression tests for GraphNodes.

"""
import anyio
import httpx
import openai
import pytest
from langgraph.errors import GraphRecursionError
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from graph.nodes import GraphNodes


def _mcp_error():
    return McpError(ErrorData(code=-1, message="connection reset"))


def _api_error(message):
    req = httpx.Request("POST", "http://test")
    return openai.APIError(message, req, body=None)


class FakeAgent:
    """agent.ainvoke() that raises/returns whatever you queue up, one item
    per call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        outcome = self._responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeReadyEvent:
    async def wait(self):
        return None


class FakeReconnectEvent:
    def __init__(self):
        self.set_calls = 0

    def set(self):
        self.set_calls += 1


class FakeChatbot:
    def __init__(self, responses):
        self.agent = FakeAgent(responses)
        self.reconnect_event = FakeReconnectEvent()
        self.ready_event = FakeReadyEvent()
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_agent(self):
        self.acquire_calls += 1
        return self.agent

    async def release_agent(self):
        self.release_calls += 1


def _state(retry_count=0):
    return {
        "original_query": "q",
        "current_query": "q",
        "messages": [],
        "cache_hit": False,
        "draft_answer": None,
        "citation_check_passed": False,
        "citation_issues": [],
        "search_retries": retry_count,
        "clarification_question": None,
        "clarification_options": [],
        "answer_is_reliable": False,
        "fetched_papers": [],
    }


SUCCESS_STATE = {"messages": ["ok"]}


async def test_success_returns_agent_state_with_none_fallback():
    node = GraphNodes(llm=None, chatbot=FakeChatbot([SUCCESS_STATE]))
    result = await node._invoke_agent_with_recovery([], _state(), {})

    assert result == (SUCCESS_STATE, None)


async def test_graph_recursion_error_returns_2tuple_with_fallback():
    node = GraphNodes(llm=None, chatbot=FakeChatbot([GraphRecursionError()]))
    agent_state, fallback = await node._invoke_agent_with_recovery([], _state(retry_count=1), {})

    assert agent_state is None
    assert fallback is not None
    assert fallback["search_retries"] == 2
    assert fallback["answer_is_reliable"] is False
    assert fallback["fetched_papers"] == []
    assert "several attempts" in fallback["draft_answer"]


async def test_mcp_reconnect_retry_succeeds():
    node = GraphNodes(llm=None, chatbot=FakeChatbot([_mcp_error(), SUCCESS_STATE]))
    result = await node._invoke_agent_with_recovery([], _state(), {})

    assert result == (SUCCESS_STATE, None)


async def test_mcp_reconnect_exhausts_retries_returns_2tuple_with_fallback():
    node = GraphNodes(llm=None, chatbot=FakeChatbot([_mcp_error(), _mcp_error(), _mcp_error()]))
    agent_state, fallback = await node._invoke_agent_with_recovery([], _state(retry_count=0), {})

    assert agent_state is None
    assert fallback is not None
    assert fallback["search_retries"] == 1
    assert "temporarily unavailable" in fallback["draft_answer"]


async def test_httpx_read_timeout_retry_succeeds():
    node = GraphNodes(llm=None, chatbot=FakeChatbot([httpx.ReadTimeout("stalled"), SUCCESS_STATE]))
    result = await node._invoke_agent_with_recovery([], _state(), {})

    assert result == (SUCCESS_STATE, None)


async def test_httpx_read_timeout_retry_fails_returns_2tuple_with_fallback():
    node = GraphNodes(
        llm=None,
        chatbot=FakeChatbot([httpx.ReadTimeout("stalled"), RuntimeError("still broken")]),
    )
    agent_state, fallback = await node._invoke_agent_with_recovery([], _state(retry_count=2), {})

    assert agent_state is None
    assert fallback is not None
    assert fallback["search_retries"] == 3
    assert "took too long" in fallback["draft_answer"]


async def test_malformed_tool_call_retry_succeeds():
    node = GraphNodes(
        llm=None,
        chatbot=FakeChatbot([_api_error("tool call validation failed: bad args"), SUCCESS_STATE]),
    )
    result = await node._invoke_agent_with_recovery([], _state(), {})

    assert result == (SUCCESS_STATE, None)


async def test_malformed_tool_call_retry_fails_returns_2tuple_with_fallback():
    node = GraphNodes(
        llm=None,
        chatbot=FakeChatbot(
            [_api_error("Failed to call a function"), RuntimeError("still broken")]
        ),
    )
    agent_state, fallback = await node._invoke_agent_with_recovery([], _state(), {})

    assert agent_state is None
    assert fallback is not None
    assert "trouble processing" in fallback["draft_answer"]


async def test_non_malformed_api_error_reraises_instead_of_being_swallowed():
    """The `else: raise` branch (line 201) -- an APIError that ISN'T a
    malformed-tool-call must propagate up to run_agent's own try/except,
    not get silently absorbed here."""
    node = GraphNodes(llm=None, chatbot=FakeChatbot([_api_error("rate limited")]))

    with pytest.raises(openai.APIError):
        await node._invoke_agent_with_recovery([], _state(), {})


@pytest.mark.parametrize(
    "responses",
    [
        [SUCCESS_STATE],
        [GraphRecursionError()],
        [_mcp_error(), _mcp_error(), _mcp_error()],
        [httpx.ReadTimeout("x"), RuntimeError("x")],
        [_api_error("tool call validation failed"), RuntimeError("x")],
        [RuntimeError("totally unexpected")],
    ],
)
async def test_every_handled_path_returns_a_real_2tuple(responses):
    """The core regression: no matter which failure mode fires, the return
    value must be unpackable as `agent_state, fallback_state = ...` without
    raising ValueError. A bare dict return (the pre-fix bug) fails this."""
    node = GraphNodes(llm=None, chatbot=FakeChatbot(list(responses)))

    result = await node._invoke_agent_with_recovery([], _state(), {})

    assert isinstance(result, tuple) and len(result) == 2
    agent_state, fallback_state = result  # must not raise
    assert (agent_state is None) != (fallback_state is None)  # exactly one is set


async def test_run_agent_survives_reraised_api_error_without_crashing():
    """End-to-end through run_agent(): a non-malformed APIError propagates
    out of _invoke_agent_with_recovery, and run_agent's own outer
    except-Exception must catch it and return a plain state dict -- not
    raise, not return a tuple."""
    node = GraphNodes(llm=None, chatbot=FakeChatbot([_api_error("rate limited")]))

    result = await node.run_agent(_state(retry_count=0), {})

    assert isinstance(result, dict)
    assert result["answer_is_reliable"] is False
    assert result["search_retries"] == 1
    assert "unexpected issue" in result["draft_answer"]