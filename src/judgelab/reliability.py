"""Per-rater reliability and item-level disagreement reports."""

from __future__ import annotations

import numpy as np

from judgelab.aggregate import AggregationResult, _loo_accuracy
from judgelab.agreement import pairwise_cohen
from judgelab.io import RatingSet


def _truth_idx(rs: RatingSet, truth: dict[str, object] | None) -> np.ndarray | None:
    if not truth:
        return None
    pos = {c: i for i, c in enumerate(rs.classes)}
    return np.array([pos.get(truth.get(item), -1) for item in rs.items])


def rater_report(
    rs: RatingSet,
    results: dict[str, AggregationResult],
    truth: dict[str, object] | None = None,
) -> list[dict]:
    """One row per rater.

    - loo_agreement: agreement with the (unweighted) majority of the other raters
    - mean_kappa: Cohen's kappa with each other rater, averaged over shared items
    - ds_accuracy: accuracy implied by the rater's Dawid-Skene confusion matrix
    - weight: the weighted-vote weight (log-odds of estimated accuracy)
    - true_accuracy: share correct against gold labels (only when truth is given)
    """
    loo = _loo_accuracy(rs, np.ones(rs.n_raters), prior_strength=0.0)
    pairs = pairwise_cohen(rs)["pairs"]
    t_idx = _truth_idx(rs, truth)
    n_ratings = np.bincount(rs.rater_idx, minlength=rs.n_raters)
    ds = results.get("dawid_skene")
    wv = results.get("weighted")

    rows = []
    for r_i, rater in enumerate(rs.raters):
        ks, ns = [], []
        for (a, b), v in pairs.items():
            if rater in (a, b) and not np.isnan(v["kappa"]):
                ks.append(v["kappa"])
                ns.append(v["n"])
        row = {
            "rater": rater,
            "type": rs.rater_types.get(rater, ""),
            "n_ratings": int(n_ratings[r_i]),
            "loo_agreement": float(loo[r_i]),
            "mean_kappa": float(np.average(ks, weights=ns)) if ks else float("nan"),
        }
        if ds is not None:
            row["ds_accuracy"] = ds.rater_accuracy[rater]
        if wv is not None:
            row["weight"] = wv.rater_weights[rater]
        if t_idx is not None:
            mask = (rs.rater_idx == r_i) & (t_idx[rs.item_idx] >= 0)
            hits = (rs.label_idx[mask] == t_idx[rs.item_idx[mask]]).sum()
            row["true_accuracy"] = float(hits / mask.sum()) if mask.any() else float("nan")
        rows.append(row)
    return rows


def disagreements(
    rs: RatingSet,
    results: dict[str, AggregationResult],
    top: int = 15,
) -> list[dict]:
    """Items where raters disagree most, ranked by majority vote share
    (lowest first) then by Dawid-Skene posterior entropy (highest first).
    Items where the majority and Dawid-Skene labels differ are flagged."""
    counts = rs.vote_counts()
    n = counts.sum(axis=1)
    share = counts.max(axis=1) / np.maximum(n, 1)
    mv = results.get("majority")
    ds = results.get("dawid_skene")
    if ds is not None:
        p = np.clip(ds.posterior, 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=1)
    else:
        entropy = np.zeros(rs.n_items)
    order = np.lexsort((-entropy, share))
    dense = rs.dense()
    rows = []
    for i in order:
        if share[i] >= 1.0 and n[i] > 1:
            continue  # unanimous
        if n[i] < 2:
            continue
        ratings = {
            rs.raters[r]: rs.classes[dense[i, r]] for r in range(rs.n_raters) if dense[i, r] >= 0
        }
        row = {
            "item_id": rs.items[i],
            "n_ratings": int(n[i]),
            "top_share": float(share[i]),
            "ratings": ratings,
        }
        if mv is not None:
            row["majority"] = mv.classes[mv.label_idx[i]]
        if ds is not None:
            row["dawid_skene"] = ds.classes[ds.label_idx[i]]
            row["ds_confidence"] = float(ds.posterior[i, ds.label_idx[i]])
        if mv is not None and ds is not None:
            row["methods_differ"] = bool(mv.label_idx[i] != ds.label_idx[i])
        rows.append(row)
        if len(rows) >= top:
            break
    return rows


def accuracy_against(result: AggregationResult, truth: dict[str, object]) -> dict[str, float]:
    """Accuracy (and mean absolute error for numeric labels) vs gold labels."""
    labels = result.as_dict()
    items = [i for i in labels if i in truth]
    if not items:
        return {"n": 0, "accuracy": float("nan")}
    acc = float(np.mean([labels[i] == truth[i] for i in items]))
    out = {"n": len(items), "accuracy": acc}
    try:
        out["mae"] = float(np.mean([abs(float(labels[i]) - float(truth[i])) for i in items]))
    except (TypeError, ValueError):
        pass
    return out
