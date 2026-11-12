"""Benchmark the aggregators against known ground truth on synthetic data."""

from __future__ import annotations

import numpy as np

from judgelab.aggregate import bayes_posterior, dawid_skene, majority_vote, weighted_vote
from judgelab.io import ratings_from_records
from judgelab.reliability import accuracy_against
from judgelab.synth import SCENARIOS, generate

METHOD_ORDER = ["majority", "weighted", "dawid_skene", "oracle"]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run_once(scenario: str, n_items: int, seed: int) -> dict:
    data = generate(scenario, n_items=n_items, seed=seed)
    rs = ratings_from_records(data.records, classes=data.classes)
    results = {
        "majority": majority_vote(rs, seed=seed),
        "weighted": weighted_vote(rs, seed=seed),
        "dawid_skene": dawid_skene(rs),
    }
    spec_by_id = {s.rater_id: s for s in data.specs}
    true_conf = np.stack([spec_by_id[r].confusion for r in rs.raters])
    post, _ = bayes_posterior(rs, true_conf, data.priors)
    oracle_idx = post.argmax(axis=1)
    oracle = {item: rs.classes[k] for item, k in zip(rs.items, oracle_idx)}

    row: dict = {"scenario": scenario, "seed": seed}
    for name, res in results.items():
        m = accuracy_against(res, data.truth)
        row[f"{name}_acc"] = m["accuracy"]
        row[f"{name}_mae"] = m["mae"]
    row["oracle_acc"] = float(np.mean([oracle[i] == data.truth[i] for i in rs.items]))
    row["oracle_mae"] = float(np.mean([abs(oracle[i] - data.truth[i]) for i in rs.items]))

    true_acc = np.array([spec_by_id[r].true_accuracy(data.priors) for r in rs.raters])
    ds_acc = np.array([results["dawid_skene"].rater_accuracy[r] for r in rs.raters])
    loo_acc = np.array([results["weighted"].rater_accuracy[r] for r in rs.raters])
    row["ds_rater_acc_mae"] = float(np.mean(np.abs(ds_acc - true_acc)))
    row["ds_rater_acc_spearman"] = _spearman(ds_acc, true_acc)
    row["loo_rater_acc_mae"] = float(np.mean(np.abs(loo_acc - true_acc)))
    row["ds_iterations"] = results["dawid_skene"].n_iter
    row["ds_converged"] = results["dawid_skene"].converged
    return row


def run_benchmark(
    scenarios: list[str] | None = None,
    n_items: int = 400,
    seeds: int = 20,
) -> dict:
    scenarios = scenarios or list(SCENARIOS)
    runs = [run_once(s, n_items, seed) for s in scenarios for seed in range(seeds)]
    summary = {}
    for s in scenarios:
        rows = [r for r in runs if r["scenario"] == s]
        agg = {"n_runs": len(rows)}
        for key in rows[0]:
            if key in ("scenario", "seed"):
                continue
            vals = np.array([float(r[key]) for r in rows])
            agg[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        ds_minus_mv = np.array([r["dawid_skene_acc"] - r["majority_acc"] for r in rows])
        agg["ds_minus_mv"] = {
            "mean": float(ds_minus_mv.mean()),
            "std": float(ds_minus_mv.std()),
            "wins": int((ds_minus_mv > 0).sum()),
            "losses": int((ds_minus_mv < 0).sum()),
        }
        summary[s] = agg
    return {
        "config": {"scenarios": scenarios, "n_items": n_items, "seeds": seeds},
        "summary": summary,
        "runs": runs,
    }


def benchmark_markdown(bench: dict) -> str:
    cfg = bench["config"]
    lines = [
        f"Synthetic benchmark: {cfg['n_items']} items x {cfg['seeds']} seeds per scenario, "
        "4-point rubric. Accuracy vs ground truth, mean (sd).",
        "",
        "| Scenario | Majority | Weighted | Dawid-Skene | Oracle (true confusions) | DS - MV (pp) | DS wins / losses |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, agg in bench["summary"].items():
        cells = [s]
        for m in METHOD_ORDER:
            v = agg[f"{m}_acc"]
            cells.append(f"{v['mean'] * 100:.1f} ({v['std'] * 100:.1f})")
        d = agg["ds_minus_mv"]
        cells.append(f"{d['mean'] * 100:+.1f}")
        cells.append(f"{d['wins']} / {d['losses']}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Scenario | MAE majority | MAE Dawid-Skene | Rater-accuracy error, DS (abs) | Rater-accuracy error, leave-one-out (abs) | Spearman(DS est, true) |",
        "|---|---|---|---|---|---|",
    ]
    for s, agg in bench["summary"].items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s,
                agg["majority_mae"]["mean"],
                agg["dawid_skene_mae"]["mean"],
                agg["ds_rater_acc_mae"]["mean"],
                agg["loo_rater_acc_mae"]["mean"],
                agg["ds_rater_acc_spearman"]["mean"],
            )
        )
    return "\n".join(lines) + "\n"
