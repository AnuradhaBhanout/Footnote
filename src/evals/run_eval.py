
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from db.rag_index import HybridIndex
import json
QUERIES = Path(__file__).parent / "queries.jsonl"

def recall_at_k(ranked_ids: list[str], expected_id: str, k: int)-> float:
    rel = 0
    if expected_id in ranked_ids[:k]:
        rel+=1
    return rel


def reciprocal_rank(ranked_ids: list[str], expected_id: str)->float:

    if expected_id in ranked_ids:
        return 1/(ranked_ids.index(expected_id)+1)
    return 0


def run_eval(index: HybridIndex, queries: list[dict], k: int = 5, alpha: float = 0.5) -> dict:
    per_query = []
    recalls = []
    rrs = []
    misses = []
    for q in queries:
        results = index.search(q['query'],top_k=k,alpha=alpha)
        ranked = [r["paper_id"] for r in results]
        expected = q["expected_paper_id"]

        recalls.append(recall_at_k(ranked,expected,k))
        
        rr = reciprocal_rank(ranked, expected)
        rrs.append(rr)
        per_query.append({"query": q["query"], "expected": expected, "rr": rr, "ranked": ranked})
        if expected not in ranked:
            misses.append({"query":q["query"],"expected":expected,"got":ranked})
    return{
        "recall_at_k":sum(recalls)/len(recalls),
        "mrr":sum(rrs)/len(rrs),
        "n":len(queries),
        "misses":misses,
        "per_query":per_query,
        }
    





if __name__ == "__main__":
    index = HybridIndex()
    index.load_cache_or_build()
    queries = [json.loads(l) for l in open(QUERIES,encoding="utf-8") if l.strip()]
    result = run_eval(index, queries)
    print(f"recall@5 {result['recall_at_k']:.3f}   MRR {result['mrr']:.3f}   n={result['n']}\n")
    for r in result["per_query"]:
        flag = "  <-- " if r["rr"] < 1.0 else "       "
        print(f"{r['rr']:.3f}{flag}{r['query'][:60]}")
        if r["rr"] < 1.0:
            print(f"         expected {r['expected']}")
            print(f"         got      {r['ranked']}")













