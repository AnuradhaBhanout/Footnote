import json
import pytest
import server.tools as tools




@pytest.fixture(autouse=True)
def _isolate_search_papers(monkeypatch,tmp_path):
    monkeypatch.setattr(tools,"PAPER_DIR",str(tmp_path))
    monkeypatch.setattr(tools,"_insert_papers_sync",lambda paper_info, topic:None)
    monkeypatch.setattr(tools._hybrid_index,"refresh_if_stale",lambda:None)

def _patch_arxiv_results(monkeypatch,papers):
    class FakeClient:
        def results(self,search):
            return list(papers)

    monkeypatch.setattr(tools.arxiv,"Client",lambda:FakeClient)
    monkeypatch.setattr(tools.arxiv,"Search",lambda **kw:object)


async def test_no_keyerror_when_evaluator_says_insufficient(monkeypatch,fake_paper_factory):
    papers = [
        fake_paper_factory("2406.04405v2","Streamlining and standardizing software citations"),
        fake_paper_factory("1808.04096v1","Directed Policy Gradient for safe RL"),
    ]
    _patch_arxiv_results(monkeypatch,papers )

    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda topic,paper_rel:{
            "sufficient":False,
            "best_paper_id":None,
            "reason":"None of the listed papers address the topic.",
        },
    )

    result = await tools.search_papers("APRS CPS 230 regulation")

    assert result["paper_ids"] == []
    assert result["sufficient"] is False



async def test_paper_rel_built_from_correct_per_id_titles(monkeypatch,fake_paper_factory):
    papers = [
        fake_paper_factory("1111.1111v1","Title One"),
        fake_paper_factory("2222.2222v1","Title Two"),
        fake_paper_factory("3333.3333v1","Title Three"),

    ]
    _patch_arxiv_results(monkeypatch,papers )

    captured = {}

    def fake_evaluate(topic,paper_rel):
        captured["paper_rel"]=paper_rel
        return {"sufficient":False,"best_paper_id":None,"reason":"n/n"}

    monkeypatch.setattr(tools,"evaluate_relevance",fake_evaluate)

    await tools.search_papers("some_topic")

    by_id = {r["paper_id"]: r["title"] for r in captured["paper_rel"]}
    assert by_id == {
        "1111.1111v1":"Title One",
        "2222.2222v1":"Title Two",
        "3333.3333v1":"Title Three",

    }

async def test_arxiv_results_before_evaluator(monkeypatch):
    _patch_arxiv_results(monkeypatch,[])
    called = {"n": 0}

    def fake_evaluate(topic,paper_rel):
        called["n"] += 1
        return {"sufficient":False,"best_paper_id":None,"reason":"n/a"}

    monkeypatch.setattr(tools,"evaluate_relvance",fake_evaluate)

    result = await tools.search_papers("nonexistent topic xyz")

    assert result == {"paper_ids":[],"sufficient":False,"reason":"arXiv returned no results."}

    assert called["n"] == 0


async def test_id_not_in_paper_ids(monkeypatch,fake_paper_factory):

    papers = [fake_paper_factory("4444.4444v1","Some Title")]
    _patch_arxiv_results(monkeypatch,papers)

    
    monkeypatch.setattr(
        tools,
        "evaluate_relevance",
        lambda topic,paper_rel:{
            "sufficient":True,
            "best_paper_id":"9999.9999v1",
            "reason":"hallucinated",
        },
    )

    result = await tools.search_papers("some topic")

    assert result["paper_ids"] == []
    assert result["sufficient"] is False