"""Markdown report for one ratings file: agreement, aggregation, reliability, disagreements."""

from __future__ import annotations

import math

from judgelab.aggregate import dawid_skene, majority_vote, weighted_vote
from judgelab.agreement import agreement_summary
from judgelab.io import RatingSet
from judgelab.reliability import accuracy_against, disagreements, rater_report


def _f(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def run_all(rs: RatingSet, gold: dict | None = None, seed: int = 0) -> dict:
    return {
        "majority": majority_vote(rs, seed=seed),
        "weighted": weighted_vote(rs, gold=gold, seed=seed),
        "dawid_skene": dawid_skene(rs),
    }


def markdown_report(rs: RatingSet, truth: dict | None = None, top: int = 15) -> str:
    results = run_all(rs)
    ag = agreement_summary(rs)
    out = [f"## Criterion: {rs.criterion}", ""]
    out.append(
        f"{rs.n_items} items, {rs.n_raters} raters, {rs.n_obs} ratings, classes {rs.classes}"
        + (f" ({rs.duplicates_dropped} duplicate ratings dropped)" if rs.duplicates_dropped else "")
    )
    out += ["", "### Inter-rater agreement", "", "| Statistic | Value |", "|---|---|"]
    for k, v in ag.items():
        if k.startswith("n_"):
            continue
        out.append(f"| {k} | {_f(v)} |")

    ds = results["dawid_skene"]
    out += [
        "",
        "### Aggregation",
        "",
        "| Method | Ties | Notes |" + (" Accuracy vs truth | MAE |" if truth else ""),
        "|---|---|---|" + ("---|---|" if truth else ""),
    ]
    for name, res in results.items():
        note = (
            f"EM {res.n_iter} iterations, converged={res.converged}"
            if name == "dawid_skene"
            else ""
        )
        row = f"| {name} | {res.n_ties} | {note} |"
        if truth:
            m = accuracy_against(res, truth)
            row += f" {_f(m['accuracy'])} | {_f(m.get('mae'))} |"
        out.append(row)
    n_diff = int((results["majority"].label_idx != ds.label_idx).sum())
    out.append("")
    out.append(f"Majority vote and Dawid-Skene disagree on {n_diff} of {rs.n_items} items.")

    rows = rater_report(rs, results, truth)
    cols = ["rater", "type", "n_ratings", "loo_agreement", "mean_kappa", "ds_accuracy", "weight"]
    if truth:
        cols.append("true_accuracy")
    out += [
        "",
        "### Per-rater reliability",
        "",
        "loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by "
        "the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in sorted(rows, key=lambda r: -r.get("ds_accuracy", 0)):
        out.append("| " + " | ".join(_f(r.get(c)) for c in cols) + " |")

    dis = disagreements(rs, results, top=top)
    out += [
        "",
        f"### Most contested items (top {len(dis)})",
        "",
        "| item | ratings | top share | majority | Dawid-Skene (conf) | labels |",
        "|---|---|---|---|---|---|",
    ]
    for d in dis:
        flag = " *" if d.get("methods_differ") else ""
        labels = ", ".join(f"{k}={v}" for k, v in d["ratings"].items())
        out.append(
            f"| {d['item_id']}{flag} | {d['n_ratings']} | {d['top_share']:.2f} | {d.get('majority')} | "
            f"{d.get('dawid_skene')} ({_f(d.get('ds_confidence'), 2)}) | {labels} |"
        )
    out.append("")
    out.append("`*` = majority vote and Dawid-Skene pick different labels.")
    return "\n".join(out) + "\n"
