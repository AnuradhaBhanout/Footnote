"""Test for hybrid_search_papers()"""

import asyncio
import pytest

import server.tools as tools


@pytest.fixture(autouse=True)
def _isolate_hybrid_search(monkeypatch):
    monkeypatch.setattr(tools,"_ensure_index_loaded",lambda:None)
   


def _fake_results():
    return[
        {"paper_id":"2601.18779v1","score":1.0,"dense_score":1.0,"bm25_score":1.0},
        {"paper_id":"2605.23493v1","score":0.85,"dense_score":0.9,"bm25_score":0.8},
    ]

async def test_evaluator_verdict_when_sufficient(monkeypatch):
    results = _fake_results()
    monkeypatch.setattr(tools._hybrid_index,"search",lambda query,top_k,alpha:results)

    monkeypatch.setattr(
        tools,
        "load_all_papers",
        lambda:{
            "2601.18779v1": {"title":"POPE:Learing to reason a hard problems"},
            "2605.23493v1":{"title":"EDGE-OPD"}

        },
    )

    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda query,results:{
            "sufficient":True,
            "best_paper_id":"2601.18779v1",
            "reason":"Directly matches",
        },
    )

    out = await tools.hybrid_search_papers("Priviledge On-Policy Exploration")

    assert out["evaluator_verdict"]["sufficient"] is True
    assert out["evaluator_verdict"]["best_paper_id"] == "2601.18779v1"
    assert [r["paper_id"] for r in out["results"]] == ["2601.18779v1","2605.23493v1"]


async def test_title_rejects_low_overlap_match(monkeypatch):
    results = _fake_results()
    monkeypatch.setattr(tools._hybrid_index,"search",lambda query,top_k,alpha:results)

    monkeypatch.setattr(
        tools,
        "load_all_papers",
        lambda:{
            "2601.18779v1": {"title":"Completely unrelated paper about cheese"},
            "2605.23493v1":{"title":"EDGE-OPD"}

        },
    )

    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda query,results:{
            "sufficient":True,
            "best_paper_id":"2601.18779v1",
            "reason":"judge think it matches",
        },
    )

    out = await tools.hybrid_search_papers('find the paper titled "Priviledge On-Policy Exploration"')

    assert out["evaluator_verdict"]["sufficient"] is False
    assert out["evaluator_verdict"]["best_paper_id"] is None
    assert "not found among retrieved papers" in out["evaluator_verdict"]["reason"]






async def test_title_accept_high_overlap_match(monkeypatch):
    results = _fake_results()
    monkeypatch.setattr(tools._hybrid_index,"search",lambda query,top_k,alpha:results)

    monkeypatch.setattr(
        tools,
        "load_all_papers",
        lambda:{
            "2601.18779v1": {"title":"POPE: Learning to reason on hard problems via Priviledge On-Policy Exploration"},
            "2605.23493v1":{"title":"EDGE-OPD"}

        },
    )

    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda query,results:{
            "sufficient":True,
            "best_paper_id":"2601.18779v1",
            "reason":"matches",
        },
    )

    out = await tools.hybrid_search_papers('find the paper titled "Priviledge On-Policy Exploration"')

    assert out["evaluator_verdict"]["sufficient"] is True
    assert out["evaluator_verdict"]["best_paper_id"]  == "2601.18779v1"



async def test_evaluate_relevance_is_wraped_to_thread(monkeypatch):
    results = _fake_results()
    monkeypatch.setattr(tools._hybrid_index,"search",lambda query,top_k,alpha:results)

    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda query,results:{
            "sufficient":False,
            "best_paper_id":None,
            "reason":"n/a",
        },
    )

    real_to_thread = asyncio.to_thread

    calls = []

    async def spy_to_thread(func,*args,**kwargs):
        calls.append(func)
        return await real_to_thread(func,*args,**kwargs)

    monkeypatch.setattr(tools.asyncio,"to_thread",spy_to_thread)

    await tools.hybrid_search_papers("some query")

    assert tools.evaluate_relevance in calls,(
        "evaluate_relevance was called directly, not via asyncio.to_thread --"
        "it will block the event loop for every concurrent request."
    )