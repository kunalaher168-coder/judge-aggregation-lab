"""Agreement statistics, cross-checked against reference implementations."""

import numpy as np
import pytest
from conftest import records_from_matrix

from judgelab.agreement import (
    agreement_summary,
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    pairwise_cohen,
    percent_agreement,
)
from judgelab.io import ratings_from_records

sklearn_metrics = pytest.importorskip("sklearn.metrics")
sm_irr = pytest.importorskip("statsmodels.stats.inter_rater")
krippendorff = pytest.importorskip("krippendorff")


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohen_matches_sklearn(rng, weights):
    for _ in range(20):
        a = rng.integers(0, 4, size=60)
        b = np.where(rng.random(60) < 0.6, a, rng.integers(0, 4, size=60))
        ours = cohen_kappa(a, b, n_classes=4, weights=weights)
        ref = sklearn_metrics.cohen_kappa_score(a, b, labels=[0, 1, 2, 3], weights=weights)
        assert ours == pytest.approx(ref, abs=1e-10)


def test_cohen_perfect_and_degenerate():
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    assert np.isnan(cohen_kappa([1, 1, 1], [1, 1, 1]))


def test_fleiss_wikipedia_example():
    # Fleiss (1971) worked example as reproduced on Wikipedia: 10 subjects, 14 raters, 5 categories.
    table = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    recs = []
    for i, row in enumerate(table):
        labels = [c + 1 for c, n in enumerate(row) for _ in range(n)]
        recs += [
            {"item_id": f"s{i}", "rater_id": f"r{j}", "label": lab} for j, lab in enumerate(labels)
        ]
    rs = ratings_from_records(recs)
    assert fleiss_kappa(rs) == pytest.approx(sm_irr.fleiss_kappa(table), abs=1e-12)
    assert fleiss_kappa(rs) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_matches_statsmodels_random(rng):
    for _ in range(10):
        truth = rng.integers(1, 5, size=50)
        mat = [
            [int(t) if rng.random() < 0.7 else int(rng.integers(1, 5)) for _ in range(6)]
            for t in truth
        ]
        rs = ratings_from_records(records_from_matrix(mat))
        table, _ = sm_irr.aggregate_raters(np.array(mat))
        assert fleiss_kappa(rs) == pytest.approx(sm_irr.fleiss_kappa(table), abs=1e-10)


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval"])
def test_krippendorff_matches_package_with_missing(rng, level):
    for _ in range(10):
        truth = rng.integers(1, 6, size=40).astype(float)
        mat = np.array(
            [
                [t if rng.random() < 0.65 else float(rng.integers(1, 6)) for _ in range(5)]
                for t in truth
            ]
        )
        mat[rng.random(mat.shape) < 0.3] = np.nan  # 30% missing
        rs = ratings_from_records(records_from_matrix(mat.tolist()))
        # krippendorff expects coders x units; classes absent from the data do not change alpha
        ref = krippendorff.alpha(reliability_data=mat.T, level_of_measurement=level)
        assert krippendorff_alpha(rs, level) == pytest.approx(ref, abs=1e-9)


def test_krippendorff_perfect_agreement():
    rs = ratings_from_records(records_from_matrix([[1, 1], [2, 2], [3, 3, 3]]))
    assert krippendorff_alpha(rs, "nominal") == pytest.approx(1.0)


def test_percent_agreement_and_summary(tiny_rs):
    # pairs: item0 1/3 agree, item1 3/3, item2 0/1, item3 1/3 -> (2+6+0+2)/(6+6+2+6)
    assert percent_agreement(tiny_rs) == pytest.approx(10 / 20)
    s = agreement_summary(tiny_rs)
    assert s["n_items"] == 4 and s["n_ratings"] == 11
    assert "krippendorff_alpha_ordinal" in s
    pc = pairwise_cohen(tiny_rs)
    assert set(pc["pairs"]) == {("r00", "r01"), ("r00", "r02"), ("r01", "r02")}
