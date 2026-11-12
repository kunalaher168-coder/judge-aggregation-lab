import json

import numpy as np
import pytest

from judgelab.agreement import cohen_kappa
from judgelab.benchmark import benchmark_markdown, run_benchmark
from judgelab.io import load_ratings, ratings_from_records
from judgelab.synth import SCENARIOS, generate


def test_load_ratings_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"item_id": "a", "rater_id": "x", "label": 2, "criterion": "c1"},
        {"item_id": "a", "rater_id": "y", "label": 3, "criterion": "c1"},
        {"item_id": "b", "rater_id": "x", "label": 1, "criterion": "c1"},
        {"item_id": "b", "rater_id": "x", "label": 2, "criterion": "c1"},  # duplicate: last wins
        {"item_id": "a", "rater_id": "x", "label": 9, "criterion": "c2"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    rs = load_ratings(p, criterion="c1")
    assert rs.items == ["a", "b"] and rs.raters == ["x", "y"]
    assert rs.classes == [2, 3]
    assert rs.duplicates_dropped == 1
    dense = rs.dense()
    assert dense.tolist() == [[0, 1], [0, -1]]
    assert load_ratings(p, criterion="c2").classes == [9]


def test_missing_key_and_bad_json(tmp_path):
    with pytest.raises(ValueError):
        ratings_from_records([{"item_id": "a", "label": 1}])
    p = tmp_path / "bad.jsonl"
    p.write_text('{"item_id": "a"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_ratings(p)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synth_is_deterministic_and_complete(scenario):
    a = generate(scenario, n_items=50, seed=4)
    b = generate(scenario, n_items=50, seed=4)
    assert a.records == b.records and a.truth == b.truth
    per_item = {}
    for r in a.records:
        per_item.setdefault(r["item_id"], 0)
        per_item[r["item_id"]] += 1
    assert len(per_item) == 50
    assert len(set(per_item.values())) == 1


def test_synth_empirical_accuracy_matches_spec():
    data = generate("independent", n_items=4000, seed=0)
    for spec in data.specs:
        mine = [r for r in data.records if r["rater_id"] == spec.rater_id]
        acc = np.mean([r["label"] == data.truth[r["item_id"]] for r in mine])
        assert acc == pytest.approx(spec.true_accuracy(data.priors), abs=0.035)


def test_correlated_group_agrees_more_than_independent_raters_would():
    data = generate("correlated_llm", n_items=2000, seed=0)
    rs = ratings_from_records(data.records, classes=data.classes)
    d = rs.dense()
    ia, ib = rs.raters.index("llm_family_a"), rs.raters.index("llm_family_b")
    k_family = cohen_kappa(d[:, ia], d[:, ib], 4)
    h1, h2 = rs.raters.index("human_1"), rs.raters.index("human_2")
    k_humans = cohen_kappa(d[:, h1], d[:, h2], 4)
    # family members are individually *less* accurate than the humans but agree more
    assert k_family > k_humans


def test_benchmark_headline_claims():
    bench = run_benchmark(["mixed", "biased_llm", "correlated_llm"], n_items=300, seeds=3)
    s = bench["summary"]
    assert s["mixed"]["dawid_skene_acc"]["mean"] > s["mixed"]["majority_acc"]["mean"] + 0.05
    assert (
        s["biased_llm"]["dawid_skene_acc"]["mean"] > s["biased_llm"]["majority_acc"]["mean"] + 0.05
    )
    # documented limitation: correlated judges break the independence assumption
    assert (
        s["correlated_llm"]["dawid_skene_acc"]["mean"] < s["correlated_llm"]["majority_acc"]["mean"]
    )
    md = benchmark_markdown(bench)
    assert "| mixed |" in md and "Oracle" in md
