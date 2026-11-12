import numpy as np
import pytest
from conftest import records_from_matrix

from judgelab.aggregate import (
    aggregate,
    bayes_posterior,
    dawid_skene,
    log_odds_weights,
    majority_vote,
    weighted_vote,
)
from judgelab.io import ratings_from_records
from judgelab.reliability import accuracy_against
from judgelab.synth import generate


def test_majority_simple(tiny_rs):
    res = majority_vote(tiny_rs, tie_break="lowest")
    assert res.as_dict() == {"i000": 1, "i001": 2, "i002": 1, "i003": 3}
    assert res.n_ties == 1  # item i002 is 1 vs 2
    assert np.allclose(res.posterior.sum(axis=1), 1.0)


def test_majority_random_ties_are_seeded():
    rs = ratings_from_records(records_from_matrix([[1, 2]] * 50))
    a = majority_vote(rs, seed=3).label_idx
    b = majority_vote(rs, seed=3).label_idx
    assert (a == b).all()
    assert 0 < a.mean() < 1  # both labels chosen somewhere


def test_log_odds_weights():
    w = log_odds_weights(np.array([0.25, 0.5, 0.9]), n_classes=4)
    assert w[0] == 0.0  # chance level for K=4 gets no weight
    assert w[1] > 0 and w[2] > w[1]


def test_weighted_vote_downweights_spammers():
    data = generate("spammers", n_items=400, seed=1)
    rs = ratings_from_records(data.records, classes=data.classes)
    res = weighted_vote(rs)
    careless = [res.rater_weights[r] for r in rs.raters if r.startswith("careless")]
    humans = [res.rater_weights[r] for r in rs.raters if r.startswith("human")]
    assert max(careless) < min(humans)


def test_weighted_vote_explicit_and_gold_weights():
    data = generate("mixed", n_items=300, seed=2)
    rs = ratings_from_records(data.records, classes=data.classes)
    explicit = weighted_vote(rs, weights={"human_1": 5.0, "llm_strong": 1.0})
    assert explicit.rater_weights["human_1"] == 5.0
    gold = dict(list(data.truth.items())[:100])
    g = weighted_vote(rs, gold=gold)
    assert g.rater_accuracy["careless_1"] < g.rater_accuracy["human_1"]


def test_dawid_skene_recovers_truth_better_than_majority():
    data = generate("mixed", n_items=500, seed=11)
    rs = ratings_from_records(data.records, classes=data.classes)
    mv = accuracy_against(majority_vote(rs), data.truth)["accuracy"]
    ds_res = dawid_skene(rs)
    ds = accuracy_against(ds_res, data.truth)["accuracy"]
    assert ds_res.converged
    assert ds > mv + 0.05
    assert np.allclose(ds_res.posterior.sum(axis=1), 1.0)
    assert np.allclose(ds_res.confusion.sum(axis=2), 1.0)
    true_acc = {s.rater_id: s.true_accuracy(data.priors) for s in data.specs}
    err = np.mean([abs(ds_res.rater_accuracy[r] - true_acc[r]) for r in rs.raters])
    assert err < 0.06


def test_dawid_skene_perfect_raters():
    mat = [[k, k, k] for k in [1, 2, 3, 1, 2, 3, 2, 2]]
    rs = ratings_from_records(records_from_matrix(mat))
    res = dawid_skene(rs, smoothing=0.01)
    assert res.labels == [row[0] for row in mat]


def test_dawid_skene_log_likelihood_non_decreasing():
    data = generate("biased_llm", n_items=300, seed=5)
    rs = ratings_from_records(data.records, classes=data.classes)
    lls = [dawid_skene(rs, max_iter=n, tol=0).log_likelihood for n in (1, 2, 5, 10, 30)]
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))


def test_oracle_posterior_upper_bound():
    data = generate("biased_llm", n_items=600, seed=8)
    rs = ratings_from_records(data.records, classes=data.classes)
    conf = np.stack([next(s for s in data.specs if s.rater_id == r).confusion for r in rs.raters])
    post, _ = bayes_posterior(rs, conf, data.priors)
    oracle = np.mean(
        [rs.classes[k] == data.truth[i] for i, k in zip(rs.items, post.argmax(axis=1))]
    )
    ds = accuracy_against(dawid_skene(rs), data.truth)["accuracy"]
    assert oracle >= ds - 0.02


def test_aggregate_dispatch_and_errors(tiny_rs):
    assert aggregate(tiny_rs, "mv").method == "majority"
    assert aggregate(tiny_rs, "ds").method == "dawid_skene"
    with pytest.raises(ValueError):
        aggregate(tiny_rs, "nope")


def test_string_labels():
    recs = records_from_matrix(
        [["pass", "pass", "fail"], ["fail", "fail", "fail"], ["pass", "fail", "pass"]]
    )
    rs = ratings_from_records(recs)
    assert rs.classes == ["fail", "pass"]
    assert dawid_skene(rs).as_dict()["i001"] == "fail"
