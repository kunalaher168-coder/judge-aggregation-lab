"""Multi-agent judging pipeline: planner -> judge panel -> aggregator.

Every model call is logged as a CallRecord (latency, tokens, cost, served
model), so the run can be re-analysed offline (see routing.py).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from judgelab.agents.backends import Backend, CallRecord
from judgelab.agents.pricing import model_for
from judgelab.agents.roles import Criterion, Verdict, judge_one, plan_criteria, reconcile
from judgelab.io import write_jsonl, write_text


@dataclass
class PipelineConfig:
    judge_tiers: list[str] = field(default_factory=lambda: ["haiku", "sonnet", "opus"])
    planner_tier: str = "opus"
    reconciler_tier: str = "opus"
    judge_effort: str = "low"
    planner_effort: str = "medium"
    reconciler_effort: str = "medium"
    reconcile: bool = True
    max_workers: int = 8
    item_fields: list[str] | None = None


@dataclass
class PipelineResult:
    criteria: list[Criterion]
    calls: list[CallRecord]
    panel: dict[tuple[str, str], list[Verdict]]
    final: dict[tuple[str, str], Verdict]
    final_source: dict[tuple[str, str], str]
    wall_clock_s: float
    config: PipelineConfig

    def item_scores(self) -> dict[str, dict]:
        out: dict[str, dict] = defaultdict(lambda: {"passed": 0, "failed": 0, "unknown": 0})
        for (item, _crit), v in self.final.items():
            key = {"pass": "passed", "fail": "failed"}.get(v.verdict or "", "unknown")
            out[item][key] += 1
        for s in out.values():
            n = s["passed"] + s["failed"] + s["unknown"]
            s["score"] = s["passed"] / n if n else 0.0
            s["all_pass"] = s["passed"] == n
        return dict(out)


def _majority(panel: list[Verdict]) -> Verdict | None:
    votes = [v for v in panel if v.verdict is not None]
    if not votes:
        return None
    counts = Counter(v.verdict for v in votes)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        # tie: trust the most confident judge
        best = max(votes, key=lambda v: v.confidence or 0.0)
        return Verdict(
            best.item_id,
            best.criterion,
            "majority",
            best.verdict,
            best.confidence,
            "tie: most confident",
        )
    winner = top[0][0]
    share = top[0][1] / len(votes)
    return Verdict(
        votes[0].item_id, votes[0].criterion, "majority", winner, share, "panel majority"
    )


def run_pipeline(
    items: list[dict],
    rubric: str,
    backend: Backend,
    config: PipelineConfig | None = None,
    log=print,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    t_start = time.perf_counter()
    calls: list[CallRecord] = []

    # 1. planner
    criteria, plan_rec = plan_criteria(
        backend, rubric, model_for(cfg.planner_tier), cfg.planner_effort
    )
    calls.append(plan_rec)
    log(
        f"planner: {len(criteria)} criteria ({', '.join(c.id for c in criteria)})"
        + (f" [note: {plan_rec.error}]" if plan_rec.error else "")
    )

    # 2. judge panel: one call per (tier, item, criterion), run concurrently
    jobs = [(tier, c, it) for tier in cfg.judge_tiers for it in items for c in criteria]
    panel: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [
            pool.submit(
                judge_one, backend, model_for(t), t, c, it, cfg.judge_effort, cfg.item_fields
            )
            for t, c, it in jobs
        ]
        for fut in futures:
            verdict, rec = fut.result()
            panel[(verdict.item_id, verdict.criterion)].append(verdict)
            calls.append(rec)
    n_err = sum(1 for r in calls if r.agent == "judge" and r.error)
    log(f"judges: {len(jobs)} calls across {len(cfg.judge_tiers)} tier(s), {n_err} errors")

    # 3. aggregator: unanimous -> accept; disagreement -> reconciler call (or majority)
    final: dict[tuple[str, str], Verdict] = {}
    source: dict[tuple[str, str], str] = {}
    to_reconcile = []
    for key, votes in panel.items():
        decided = {v.verdict for v in votes if v.verdict is not None}
        if len(decided) == 1:
            v = next(x for x in votes if x.verdict is not None)
            final[key] = Verdict(key[0], key[1], "unanimous", v.verdict, 1.0, "all judges agree")
            source[key] = "unanimous"
        elif cfg.reconcile and len(cfg.judge_tiers) > 1 and decided:
            to_reconcile.append(key)
        else:
            m = _majority(votes)
            final[key] = m or Verdict(key[0], key[1], "none", None, None, "no verdict")
            source[key] = "majority" if m else "no_verdict"

    by_id = {c.id: c for c in criteria}
    by_item = {it["item_id"]: it for it in items}
    if to_reconcile:
        rmodel = model_for(cfg.reconciler_tier)
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                key: pool.submit(
                    reconcile,
                    backend,
                    rmodel,
                    by_id[key[1]],
                    by_item[key[0]],
                    panel[key],
                    cfg.reconciler_effort,
                    cfg.item_fields,
                )
                for key in to_reconcile
            }
            for key, fut in futures.items():
                verdict, rec = fut.result()
                calls.append(rec)
                if verdict.verdict is None:
                    final[key] = _majority(panel[key])
                    source[key] = "majority_after_reconciler_error"
                else:
                    final[key] = verdict
                    source[key] = "reconciled"
    log(f"aggregator: {len(final)} verdicts, {len(to_reconcile)} sent to the reconciler")

    return PipelineResult(
        criteria, calls, dict(panel), final, source, time.perf_counter() - t_start, cfg
    )


def write_outputs(result: PipelineResult, out_dir: str | Path) -> None:
    """criteria.json, calls.jsonl, panel_ratings.jsonl (judgelab rating format),
    verdicts.jsonl (final) and item_scores.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_text(
        out / "criteria.json", json.dumps([asdict(c) for c in result.criteria], indent=2) + "\n"
    )
    write_jsonl(out / "calls.jsonl", (r.to_dict() for r in result.calls))
    write_jsonl(
        out / "panel_ratings.jsonl",
        (
            {
                "item_id": item,
                "criterion": crit,
                "rater_id": v.judge,
                "rater_type": "llm",
                "label": v.verdict,
                "confidence": v.confidence,
            }
            for (item, crit), votes in sorted(result.panel.items())
            for v in votes
            if v.verdict is not None
        ),
    )
    write_jsonl(
        out / "verdicts.jsonl",
        (
            {
                "item_id": item,
                "criterion": crit,
                "label": v.verdict,
                "source": result.final_source[(item, crit)],
                "confidence": v.confidence,
                "rationale": v.rationale,
            }
            for (item, crit), v in sorted(result.final.items())
        ),
    )
    write_text(
        out / "item_scores.json", json.dumps(result.item_scores(), indent=2, sort_keys=True) + "\n"
    )
    meta = {"config": asdict(result.config), "wall_clock_s": round(result.wall_clock_s, 3)}
    write_text(out / "run_meta.json", json.dumps(meta, indent=2) + "\n")
