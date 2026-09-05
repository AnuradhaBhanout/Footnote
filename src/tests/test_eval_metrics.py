from evals.run_eval import recall_at_k,reciprocal_rank


def test_recall_hit_at_first_position():
    assert recall_at_k(["target", "b", "c"], "target", 5) == 1.0

def test_recall_hit_at_position_k():
    assert recall_at_k(["a", "b", "c", "d", "target"], "target", 5) == 1.0

def test_recall_miss_beyond_k():
    assert recall_at_k(["a", "b", "c", "d", "e", "target"], "target", 5) == 0.0

def test_recall_absent():
    assert recall_at_k(["a", "b", "c"], "target", 5) == 0.0



def test_rr_first_position():
    assert reciprocal_rank(["target", "b", "c"], "target") == 1.0

def test_rr_third_position():
    assert reciprocal_rank(["a", "b", "target"], "target") == 1/3

def test_rr_absent():
    assert reciprocal_rank(["a", "b", "c"], "target") == 0.0





