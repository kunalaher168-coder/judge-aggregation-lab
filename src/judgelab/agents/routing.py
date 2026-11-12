"""Quality vs cost vs latency across model tiers, and routing policies.

Every judge call is logged per (tier, item, criterion), so routing policies
(cascades, agreement checks, panels) can be replayed exactly from one run:
a policy's cost and latency are the sums of the logged calls it would have
made, and its verdict is what those calls returned. No extra API calls.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from judgelab.aggregate import dawid_skene, majority_vote
from judgelab.io import ratings_from_records

Key = tuple[str, str]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _index(calls: list[dict]) -> dict[Key, dict[str, dict]]:
    out: dict[Key, dict[str, dict]] = defaultdict(dict)
    for c in calls:
        if c["agent"] == "judge":
            out[(c["item_id"], c["criterion"])][c["tier"]] = c
    return out


def _verdict(call: dict | None) -> str | None:
    if not call or call.get("error"):
        return None
    return call.get("output", {}).get("verdict")


def _conf(call: dict | None) -> float:
    if not call or call.get("error"):
        return 0.0
    return float(call.get("output", {}).get("confidence", 0.0))


def tier_table(calls: list[dict], truth: dict[Key, str]) -> list[dict]:
    rows = []
    tiers = sorted(
        {c["tier"] for c in calls if c["agent"] == "judge"},
        key=lambda t: (
            ["haiku", "sonnet", "opus"].index(t) if t in ("haiku", "sonnet", "opus") else 9
        ),
    )
    for t in tiers:
        tc = [
            c
            for c in calls
            if c["agent"] == "judge" and c["tier"] == t and (c["item_id"], c["criterion"]) in truth
        ]
        lat = np.array([c["latency_s"] for c in tc if not c.get("error")])
        hits = sum(_verdict(c) == truth[(c["item_id"], c["criterion"])] for c in tc)
        per_crit: dict[str, list[bool]] = defaultdict(list)
        for c in tc:
            per_crit[c["criterion"]].append(_verdict(c) == truth[(c["item_id"], c["criterion"])])
        lo, hi = wilson(hits, len(tc))
        served_other = sum(
            1 for c in tc if c.get("served_model") and c["served_model"] != c["model"]
        )
        rows.append(
            {
                "tier": t,
                "model": tc[0]["model"] if tc else "",
                "n": len(tc),
                "errors": sum(1 for c in tc if c.get("error")),
                "served_by_fallback": served_other,
                "accuracy": hits / len(tc) if tc else float("nan"),
                "ci95": (lo, hi),
                "per_criterion": {k: float(np.mean(v)) for k, v in sorted(per_crit.items())},
                "latency_mean_s": float(lat.mean()) if lat.size else float("nan"),
                "latency_p50_s": float(np.percentile(lat, 50)) if lat.size else float("nan"),
                "latency_p95_s": float(np.percentile(lat, 95)) if lat.size else float("nan"),
                "cost_usd": float(sum(c["cost_usd"] for c in tc)),
                "output_tokens_mean": float(np.mean([c["output_tokens"] for c in tc]))
                if tc
                else float("nan"),
            }
        )
    return rows


Decision = tuple[str | None, float, float, bool]  # verdict, cost, latency, escalated


def _single(t: str) -> Callable[[dict[str, dict]], Decision]:
    def f(cs):
        c = cs.get(t)
        return _verdict(c), (c or {}).get("cost_usd", 0.0), (c or {}).get("latency_s", 0.0), False

    return f


def _cascade(first: str, second: str, threshold: float) -> Callable[[dict[str, dict]], Decision]:
    def f(cs):
        a, b = cs.get(first), cs.get(second)
        cost, lat = (a or {}).get("cost_usd", 0.0), (a or {}).get("latency_s", 0.0)
        if _verdict(a) is not None and _conf(a) >= threshold:
            return _verdict(a), cost, lat, False
        return (
            _verdict(b),
            cost + (b or {}).get("cost_usd", 0.0),
            lat + (b or {}).get("latency_s", 0.0),
            True,
        )

    return f


def _agree_or_escalate(a_t: str, b_t: str, c_t: str) -> Callable[[dict[str, dict]], Decision]:
    def f(cs):
        a, b, c = cs.get(a_t), cs.get(b_t), cs.get(c_t)
        cost = (a or {}).get("cost_usd", 0.0) + (b or {}).get("cost_usd", 0.0)
        lat = max((a or {}).get("latency_s", 0.0), (b or {}).get("latency_s", 0.0))
        va, vb = _verdict(a), _verdict(b)
        if va is not None and va == vb:
            return va, cost, lat, False
        return (
            _verdict(c),
            cost + (c or {}).get("cost_usd", 0.0),
            lat + (c or {}).get("latency_s", 0.0),
            True,
        )

    return f


def _panel_cost(cs: dict[str, dict], tiers: list[str]) -> tuple[float, float]:
    return (
        sum((cs.get(t) or {}).get("cost_usd", 0.0) for t in tiers),
        max([(cs.get(t) or {}).get("latency_s", 0.0) for t in tiers] or [0.0]),
    )


def _score(name: str, decisions: dict[Key, Decision], truth: dict[Key, str]) -> dict:
    keys = [k for k in decisions if k in truth]
    hits = sum(decisions[k][0] == truth[k] for k in keys)
    lo, hi = wilson(hits, len(keys))
    return {
        "policy": name,
        "n": len(keys),
        "accuracy": hits / len(keys) if keys else float("nan"),
        "ci95": (lo, hi),
        "cost_usd": float(sum(decisions[k][1] for k in keys)),
        "latency_mean_s": float(np.mean([decisions[k][2] for k in keys])) if keys else float("nan"),
        "escalation_rate": float(np.mean([decisions[k][3] for k in keys])) if keys else 0.0,
    }


def routing_policies(
    calls: list[dict], truth: dict[Key, str], final: dict[Key, dict] | None = None
) -> list[dict]:
    idx = _index(calls)
    tiers = sorted(
        {t for cs in idx.values() for t in cs},
        key=lambda t: (
            ["haiku", "sonnet", "opus"].index(t) if t in ("haiku", "sonnet", "opus") else 9
        ),
    )
    policies: dict[str, Callable] = {f"{t} only": _single(t) for t in tiers}
    if {"haiku", "opus"} <= set(tiers):
        for th in (0.8, 0.9, 0.95):
            policies[f"haiku -> opus if haiku confidence < {th}"] = _cascade("haiku", "opus", th)
    if {"haiku", "sonnet"} <= set(tiers):
        policies["haiku -> sonnet if haiku confidence < 0.9"] = _cascade("haiku", "sonnet", 0.9)
    if {"sonnet", "opus"} <= set(tiers):
        policies["sonnet -> opus if sonnet confidence < 0.9"] = _cascade("sonnet", "opus", 0.9)
    if {"haiku", "sonnet", "opus"} <= set(tiers):
        policies["haiku + sonnet, opus only on disagreement"] = _agree_or_escalate(
            "haiku", "sonnet", "opus"
        )

    out = []
    for name, fn in policies.items():
        out.append(_score(name, {k: fn(cs) for k, cs in idx.items()}, truth))

    if len(tiers) >= 3:
        # judgement aggregation over the tier panel, pooled across criteria
        records = [
            {"item_id": f"{k[0]}::{k[1]}", "rater_id": t, "label": _verdict(c)}
            for k, cs in idx.items()
            for t, c in cs.items()
            if _verdict(c) is not None
        ]
        rs = ratings_from_records(records)
        for label, res in (
            ("panel majority vote (all tiers)", majority_vote(rs)),
            ("panel Dawid-Skene (all tiers)", dawid_skene(rs)),
        ):
            labels = res.as_dict()
            dec = {}
            for k, cs in idx.items():
                cost, lat = _panel_cost(cs, tiers)
                dec[k] = (labels.get(f"{k[0]}::{k[1]}"), cost, lat, False)
            out.append(_score(label, dec, truth))

    if final:
        rec_calls = {(c["item_id"], c["criterion"]): c for c in calls if c["agent"] == "reconciler"}
        dec = {}
        for k, cs in idx.items():
            cost, lat = _panel_cost(cs, tiers)
            esc = k in rec_calls
            if esc:
                cost += rec_calls[k].get("cost_usd", 0.0)
                lat += rec_calls[k].get("latency_s", 0.0)
            dec[k] = ((final.get(k) or {}).get("label"), cost, lat, esc)
        out.append(_score("pipeline: panel + aggregator agent", dec, truth))
    return out


def recommend(policies: list[dict], tolerance: float = 0.015) -> dict:
    best = max(p["accuracy"] for p in policies)
    ok = [p for p in policies if p["accuracy"] >= best - tolerance]
    return min(ok, key=lambda p: p["cost_usd"])


def routing_markdown(tiers: list[dict], policies: list[dict], note: str = "") -> str:
    lines = []
    if note:
        lines += [note, ""]
    lines += [
        "| Tier | Model | Judgements | Accuracy (95% CI) | Errors | Mean latency (s) | p95 latency (s) | Cost (USD) | Cost per 100 judgements |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in tiers:
        lo, hi = r["ci95"]
        lines.append(
            f"| {r['tier']} | {r['model']} | {r['n']} | {r['accuracy'] * 100:.1f}% ({lo * 100:.0f}-{hi * 100:.0f}) | "
            f"{r['errors']} | {r['latency_mean_s']:.2f} | {r['latency_p95_s']:.2f} | {r['cost_usd']:.4f} | "
            f"{r['cost_usd'] / max(r['n'], 1) * 100:.4f} |"
        )
    crit_names = sorted({c for r in tiers for c in r["per_criterion"]})
    if crit_names:
        lines += [
            "",
            "| Tier | " + " | ".join(crit_names) + " |",
            "|---|" + "---|" * len(crit_names),
        ]
        for r in tiers:
            lines.append(
                f"| {r['tier']} | "
                + " | ".join(
                    f"{r['per_criterion'].get(c, float('nan')) * 100:.0f}%" for c in crit_names
                )
                + " |"
            )
    ref = next((p for p in policies if p["policy"] == "opus only"), None)
    lines += [
        "",
        "| Routing policy | Accuracy (95% CI) | Cost (USD) | Cost vs opus only | Mean latency per judgement (s) | Escalated |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(policies, key=lambda p: p["cost_usd"]):
        lo, hi = p["ci95"]
        rel = (
            f"{p['cost_usd'] / ref['cost_usd'] * 100:.0f}%" if ref and ref["cost_usd"] > 0 else "-"
        )
        lines.append(
            f"| {p['policy']} | {p['accuracy'] * 100:.1f}% ({lo * 100:.0f}-{hi * 100:.0f}) | {p['cost_usd']:.4f} | {rel} | "
            f"{p['latency_mean_s']:.2f} | {p['escalation_rate'] * 100:.0f}% |"
        )
    rec = recommend(policies)
    lines += [
        "",
        f"Cheapest policy within 1.5 pp of the most accurate: **{rec['policy']}** "
        f"({rec['accuracy'] * 100:.1f}% at ${rec['cost_usd']:.4f}).",
    ]
    return "\n".join(lines) + "\n"
