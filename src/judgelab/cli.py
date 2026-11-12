"""Command-line interface: `judgelab <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from judgelab import __version__
from judgelab.aggregate import aggregate
from judgelab.agreement import agreement_summary
from judgelab.io import (
    criteria_in,
    load_truth,
    ratings_from_records,
    read_jsonl,
    write_jsonl,
    write_text,
)


def _criteria(rows: list[dict], wanted: str | None) -> list[str]:
    return [wanted] if wanted else criteria_in(rows)


def cmd_synth(a) -> int:
    from judgelab.synth import generate

    data = generate(a.scenario, n_items=a.items, raters_per_item=a.raters_per_item, seed=a.seed)
    write_jsonl(a.out, data.records)
    if a.truth:
        write_jsonl(a.truth, ({"item_id": k, "label": v} for k, v in data.truth.items()))
    if a.meta:
        write_text(a.meta, json.dumps(data.meta, indent=2) + "\n")
    print(f"wrote {len(data.records)} ratings for {a.items} items ({a.scenario}) to {a.out}")
    return 0


def cmd_aggregate(a) -> int:
    rows = read_jsonl(a.ratings)
    gold = load_truth(a.gold) if a.gold else None
    out_rows = []
    for crit in _criteria(rows, a.criterion):
        rs = ratings_from_records(rows, criterion=crit)
        kwargs = {"gold": gold} if (gold and a.method in ("weighted", "wv")) else {}
        res = aggregate(rs, a.method, **kwargs)
        conf = res.confidence()
        for i, item in enumerate(res.items):
            out_rows.append(
                {
                    "item_id": item,
                    "criterion": crit,
                    "label": res.classes[res.label_idx[i]],
                    "confidence": round(float(conf[i]), 4),
                    "method": res.method,
                }
            )
        print(f"[{crit}] {res.method}: {rs.n_items} items, {res.n_ties} ties", file=sys.stderr)
    if a.out:
        write_jsonl(a.out, out_rows)
        print(f"wrote {len(out_rows)} labels to {a.out}", file=sys.stderr)
    else:
        for r in out_rows:
            print(json.dumps(r))
    return 0


def cmd_agreement(a) -> int:
    rows = read_jsonl(a.ratings)
    result = {
        crit: agreement_summary(ratings_from_records(rows, criterion=crit))
        for crit in _criteria(rows, a.criterion)
    }
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        for crit, stats in result.items():
            print(f"[{crit}]")
            for k, v in stats.items():
                print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")
    return 0


def cmd_report(a) -> int:
    from judgelab.report import markdown_report

    rows = read_jsonl(a.ratings)
    truth_rows = read_jsonl(a.truth) if a.truth else []
    per_criterion_truth = any("criterion" in r for r in truth_rows)
    parts = [f"# Judgement aggregation report: {Path(a.ratings).name}", ""]
    for crit in _criteria(rows, a.criterion):
        rs = ratings_from_records(rows, criterion=crit)
        truth = {
            str(r["item_id"]): r["label"]
            for r in truth_rows
            if not per_criterion_truth or str(r.get("criterion")) == crit
        } or None
        parts.append(markdown_report(rs, truth, top=a.top))
    text = "\n".join(parts)
    if a.out:
        write_text(a.out, text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


def cmd_benchmark(a) -> int:
    from judgelab.benchmark import benchmark_markdown, run_benchmark

    bench = run_benchmark(a.scenarios or None, n_items=a.items, seeds=a.seeds)
    md = benchmark_markdown(bench)
    print(md)
    if a.out_md:
        write_text(a.out_md, md)
    if a.out_json:
        write_text(a.out_json, json.dumps(bench, indent=2) + "\n")
    return 0


def cmd_make_judge_data(a) -> int:
    from judgelab.agents.dataset import RUBRIC, generate_support_items

    items, truth = generate_support_items(a.n, seed=a.seed, defect_rate=a.defect_rate)
    out = Path(a.out_dir)
    write_jsonl(out / "items.jsonl", items)
    write_jsonl(out / "truth.jsonl", truth)
    write_text(out / "rubric.md", RUBRIC)
    n_fail = sum(1 for t in truth if t["label"] == "fail")
    print(f"wrote {len(items)} items, {len(truth)} criterion labels ({n_fail} fail) to {out}")
    return 0


def _estimate_cost(n_items: int, n_criteria: int, tiers: list[str], reconcile: bool) -> float:
    """Rough upper estimate: ~1,000 input and ~800 output tokens per judge call,
    reconciler on 35% of judgements at ~1,800 in / ~1,500 out."""
    from judgelab.agents.pricing import call_cost, model_for

    n = n_items * n_criteria
    total = sum(n * call_cost(model_for(t), 1000, 800) for t in tiers)
    total += call_cost(model_for("opus"), 1500, 3000)  # planner
    if reconcile and len(tiers) > 1:
        total += 0.35 * n * call_cost(model_for("opus"), 1800, 1500)
    return total


def cmd_judge(a) -> int:
    from judgelab.agents.backends import AnthropicBackend, SimulatedBackend
    from judgelab.agents.pipeline import PipelineConfig, run_pipeline, write_outputs
    from judgelab.agents.roles import _BRACKET_ID
    from judgelab.agents.routing import routing_markdown, routing_policies, tier_table

    items = read_jsonl(a.items)
    if a.limit is not None:
        if a.limit < 1:
            print("--limit must be at least 1", file=sys.stderr)
            return 2
        items = items[: a.limit]
    rubric = Path(a.rubric).read_text(encoding="utf-8")
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    truth_rows = read_jsonl(a.truth) if a.truth else []
    item_ids = {it["item_id"] for it in items}
    truth = {
        (r["item_id"], r["criterion"]): r["label"] for r in truth_rows if r["item_id"] in item_ids
    }

    if a.backend == "simulated":
        backend = SimulatedBackend(truth=truth, seed=a.seed)
    else:
        n_crit = len(_BRACKET_ID.findall(rubric)) or 4
        est = _estimate_cost(len(items), n_crit, tiers, not a.no_reconcile)
        print(
            f"conservative spend estimate: ${est:.2f} for {len(items)} items x {n_crit} criteria x {len(tiers)} tier(s)"
        )
        if not a.confirm_spend:
            print("refusing to call the API without --confirm-spend", file=sys.stderr)
            return 2
        backend = AnthropicBackend(use_fallbacks=not a.no_fallbacks)

    cfg = PipelineConfig(
        judge_tiers=tiers,
        planner_tier=a.planner,
        reconciler_tier=a.reconciler,
        judge_effort=a.effort,
        reconcile=not a.no_reconcile,
        max_workers=a.workers,
    )
    result = run_pipeline(items, rubric, backend, cfg)
    write_outputs(result, a.out_dir)
    total = sum(c.cost_usd for c in result.calls)
    print(
        f"done in {result.wall_clock_s:.1f}s wall clock, {len(result.calls)} calls, ${total:.4f} total"
    )

    if truth:
        calls = [c.to_dict() for c in result.calls]
        final = {k: {"label": v.verdict} for k, v in result.final.items()}
        if backend.name == "simulated":
            note = (
                f"SIMULATED backend (test double): {len(items)} items x {len(result.criteria)} criteria. "
                "Verdicts are the known truth flipped at fixed per-tier error rates; latency and token "
                "counts are drawn from assumed per-tier profiles; dollar figures apply real list prices "
                "to those simulated tokens. This checks the pipeline and the routing maths, it is not a "
                "measurement of any model."
            )
        else:
            note = (
                f"Backend: {backend.name}. {len(items)} items x {len(result.criteria)} criteria. "
                f"Wall clock {result.wall_clock_s:.1f}s with {a.workers} parallel workers; "
                f"total spend ${total:.4f} (planner + judges + aggregator)."
            )
        md = routing_markdown(tier_table(calls, truth), routing_policies(calls, truth, final), note)
        write_text(Path(a.out_dir, "routing.md"), md)
        print(md)
    return 0


def cmd_routing(a) -> int:
    from judgelab.agents.routing import routing_markdown, routing_policies, tier_table

    run = Path(a.run_dir)
    calls = read_jsonl(run / "calls.jsonl")
    final = (
        {(r["item_id"], r["criterion"]): r for r in read_jsonl(run / "verdicts.jsonl")}
        if (run / "verdicts.jsonl").exists()
        else None
    )
    truth = {(r["item_id"], r["criterion"]): r["label"] for r in read_jsonl(a.truth)}
    total = sum(c["cost_usd"] for c in calls)
    note = f"{len(calls)} logged calls, total spend ${total:.4f} (planner + judges + aggregator)."
    md = routing_markdown(tier_table(calls, truth), routing_policies(calls, truth, final), note)
    print(md)
    if a.out:
        write_text(a.out, md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judgelab", description=__doc__)
    p.add_argument("--version", action="version", version=f"judgelab {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic ratings file with known ground truth")
    s.add_argument(
        "--scenario",
        default="mixed",
        choices=["independent", "spammers", "biased_llm", "mixed", "correlated_llm"],
    )
    s.add_argument("--items", type=int, default=400)
    s.add_argument("--raters-per-item", type=int, default=-1, help="-1 = scenario default")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--truth")
    s.add_argument("--meta")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("aggregate", help="infer one label per item")
    s.add_argument("ratings")
    s.add_argument(
        "--method",
        default="dawid_skene",
        choices=["majority", "weighted", "dawid_skene", "mv", "wv", "ds"],
    )
    s.add_argument("--gold", help="gold labels JSONL used to weight raters (weighted method)")
    s.add_argument("--criterion")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_aggregate)

    s = sub.add_parser("agreement", help="Cohen / Fleiss kappa and Krippendorff alpha")
    s.add_argument("ratings")
    s.add_argument("--criterion")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_agreement)

    s = sub.add_parser(
        "report", help="markdown report: agreement, methods, rater reliability, disagreements"
    )
    s.add_argument("ratings")
    s.add_argument("--truth")
    s.add_argument("--criterion")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser(
        "benchmark", help="compare aggregators against ground truth on synthetic scenarios"
    )
    s.add_argument("--seeds", type=int, default=20)
    s.add_argument("--items", type=int, default=400)
    s.add_argument("--scenarios", nargs="*")
    s.add_argument("--out-md")
    s.add_argument("--out-json")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser(
        "make-judge-data", help="generate the synthetic support-reply dataset and rubric"
    )
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--defect-rate", type=float, default=0.35)
    s.add_argument("--out-dir", default="data/support")
    s.set_defaults(fn=cmd_make_judge_data)

    s = sub.add_parser("judge", help="run the multi-agent judging pipeline")
    s.add_argument("--items", required=True)
    s.add_argument("--rubric", required=True)
    s.add_argument(
        "--truth", help="per-criterion truth JSONL (needed by the simulator and for scoring)"
    )
    s.add_argument("--backend", default="simulated", choices=["simulated", "anthropic"])
    s.add_argument("--tiers", default="haiku,sonnet,opus")
    s.add_argument("--planner", default="opus")
    s.add_argument("--reconciler", default="opus")
    s.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, help="only the first N items")
    s.add_argument("--no-reconcile", action="store_true")
    s.add_argument(
        "--no-fallbacks", action="store_true", help="do not request server-side refusal fallbacks"
    )
    s.add_argument("--confirm-spend", action="store_true", help="required for --backend anthropic")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", default="results/run")
    s.set_defaults(fn=cmd_judge)

    s = sub.add_parser(
        "routing", help="tier quality / cost / latency and routing policies from a run"
    )
    s.add_argument("run_dir")
    s.add_argument("--truth", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_routing)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
