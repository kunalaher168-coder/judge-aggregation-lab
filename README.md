# judge-aggregation-lab

[![CI](https://github.com/kunalaher168-coder/judge-aggregation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/kunalaher168-coder/judge-aggregation-lab/actions/workflows/ci.yml)

Tools for the two questions every human or LLM evaluation pipeline runs into:

1. **Several raters scored the same items and disagree. What is the right label, and which raters can be trusted?**
   `judgelab` aggregates ratings with majority vote, reliability-weighted vote and Dawid-Skene EM, measures inter-rater agreement (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha), reports per-rater reliability and lists the items raters disagree on most.
2. **How do you run an LLM-as-judge panel without overpaying?**
   A small multi-agent pipeline (planner -> one judge agent per criterion -> aggregator agent) runs on the Anthropic Claude API, logs latency, tokens and cost for every call, and replays the log to compare quality, cost and latency across model tiers and routing policies (cascades, agreement checks, panels).

All data is synthetic, with known ground truth, and everything except the live Claude run works offline. No real or third-party data is used.

## Quick start

```bash
pip install -e ".[dev,llm]"          # Python 3.10+
pytest -q                             # 47 tests, no API key needed

# 1. judgement aggregation on a synthetic rating set with known truth
judgelab synth --scenario mixed --items 400 --out data/demo/ratings.jsonl --truth data/demo/truth.jsonl
judgelab report data/demo/ratings.jsonl --truth data/demo/truth.jsonl --out results/demo_report.md
judgelab aggregate data/demo/ratings.jsonl --method dawid_skene --out labels.jsonl
judgelab agreement data/demo/ratings.jsonl
judgelab benchmark --seeds 20 --out-md results/benchmark.md

# 2. multi-agent judging (simulated backend: free, offline)
judgelab make-judge-data --n 24 --out-dir data/support
judgelab judge --items data/support/items.jsonl --rubric data/support/rubric.md \
    --truth data/support/truth.jsonl --backend simulated --out-dir results/simulated

# 2b. the same pipeline on the real Claude API (needs ANTHROPIC_API_KEY; prints a spend estimate first)
judgelab judge --items data/support/items.jsonl --rubric data/support/rubric.md \
    --truth data/support/truth.jsonl --backend anthropic --confirm-spend --out-dir results/live
judgelab routing results/live --truth data/support/truth.jsonl
```

Docker: `docker build -t judgelab . && docker run --rm judgelab` runs the test suite in a clean image.

## Input format

One JSON object per line, one rating per line. Raters can be humans, LLM judges or both; an item does not need a rating from every rater.

```json
{"item_id": "q17", "rater_id": "judge_haiku", "rater_type": "llm", "criterion": "accuracy", "label": 3}
```

`criterion` is optional (default `overall`); each criterion is aggregated separately. Labels can be numbers (ordinal and interval statistics are then also reported) or strings such as `pass` / `fail`. Gold labels use the same shape without `rater_id`.

## Methods

| Method | What it does | Where |
|---|---|---|
| Majority vote | Plurality label; seeded random tie-break; vote share as confidence | `aggregate.majority_vote` |
| Weighted vote | Each rater's vote weighted by `log((K-1) a / (1-a))`, the optimal weight under a symmetric rater model. Accuracy `a` comes from gold items if supplied, otherwise from agreement with the leave-one-out weighted vote of the other raters (iterated) | `aggregate.weighted_vote` |
| Dawid-Skene | EM over a full confusion matrix per rater plus class priors (Dawid and Skene, 1979), MAP smoothing, majority-vote initialisation. Corrects systematic bias (e.g. a judge that scores everything one point high), not just noise | `aggregate.dawid_skene` |
| Cohen's kappa | Pairwise, unweighted or linear / quadratic weighted, on shared items | `agreement.cohen_kappa`, `pairwise_cohen` |
| Fleiss' kappa | Multi-rater; generalised to a variable number of raters per item | `agreement.fleiss_kappa` |
| Krippendorff's alpha | Coincidence-matrix form; nominal, ordinal, interval; handles missing ratings | `agreement.krippendorff_alpha` |

The agreement statistics are written from the definitions and checked in the tests against scikit-learn (`cohen_kappa_score`), statsmodels (`fleiss_kappa`, including the Fleiss 1971 worked example, kappa = 0.210) and the `krippendorff` package, with missing data, to 1e-9.

## Results 1: aggregation recovers the truth better than majority vote

Synthetic benchmark (`judgelab benchmark --seeds 20`, file `results/benchmark.md`; CI regenerates it and fails if it changes). 400 items per run, 4-point rubric, each item rated by 5 raters drawn from a pool. Every rater is defined by an explicit confusion matrix, so the truth and each rater's real accuracy are known. "Oracle" applies the true confusion matrices (Bayes-optimal), an upper bound no method can beat on average.

| Scenario | Rater pool | Majority | Weighted | Dawid-Skene | Oracle | DS - MV (pp) | DS wins / losses (20 seeds) |
|---|---|---|---|---|---|---|---|
| independent | 7 humans, noisy but unbiased | 86.5 | 87.7 | 87.8 | 89.6 | +1.3 | 15 / 4 |
| spammers | 4 humans + 3 careless raters answering at random | 68.2 | **79.8** | 76.8 | 81.8 | +8.7 | 20 / 0 |
| biased_llm | 3 humans + 4 LLM judges (2 lenient, 1 harsh, 1 strong) | 78.2 | 82.9 | **90.5** | 92.2 | +12.2 | 20 / 0 |
| mixed | all of the above, 9 raters | 74.0 | 80.5 | **86.7** | 89.6 | +12.7 | 20 / 0 |
| correlated_llm | 2 humans + 3 LLM judges that share 70% of their errors | **71.9** | 69.1 | 66.2 | 76.9 | -5.7 | 0 / 20 |

Accuracy in %, mean over 20 seeds (standard deviations 1.4 to 3.3 pp, in `results/benchmark.md`).

How Dawid-Skene estimates each rater's reliability, in the same runs:

| Scenario | Rater-accuracy error, Dawid-Skene | Rater-accuracy error, leave-one-out agreement | Rank correlation (DS estimate vs truth) |
|---|---|---|---|
| independent | 0.028 | 0.085 | 0.88 |
| spammers | 0.034 | 0.090 | 0.90 |
| biased_llm | 0.030 | 0.099 | 0.94 |
| mixed | 0.034 | 0.097 | 0.92 |
| correlated_llm | 0.204 | 0.163 | -0.70 |

What the numbers say:

- **Biased judges are where Dawid-Skene pays off** (+12 pp over majority vote, within 2-3 pp of the oracle). A lenient judge's "4" is evidence for a true 3; a confusion matrix can learn that, a single weight cannot. Weighted vote only down-weights such judges (+4.7 pp).
- **Pure random raters are handled at least as well by weighted vote** (79.8% vs 76.8%). Dawid-Skene spends parameters on confusion matrices that carry no information. Use it when bias is plausible, not by default.
- **With independent, unbiased raters all methods are close** (+1.3 pp): aggregation cannot create information that is not there.
- **Correlated judges break Dawid-Skene** (0 wins in 20 seeds, and its reliability ranking inverts). Its model assumes raters err independently given the truth; three judges from one model family that agree on the same mistakes look like three highly reliable raters. This is the practical case of several LLM judges sharing a base model or prompt.

The per-rater view from one run of the `mixed` scenario (`results/demo_report.md`):

| Rater | Type | Leave-one-out agreement | Mean kappa | Dawid-Skene accuracy | Vote weight | True accuracy |
|---|---|---|---|---|---|---|
| human_1 | human | 0.596 | 0.283 | 0.810 | 1.63 | 0.834 |
| llm_strong | llm | 0.563 | 0.302 | 0.757 | 1.62 | 0.807 |
| human_2 | human | 0.491 | 0.215 | 0.622 | 1.19 | 0.645 |
| llm_lenient_a | llm | 0.430 | 0.173 | 0.510 | 1.00 | 0.507 |
| llm_harsh | llm | 0.413 | 0.169 | 0.506 | 0.87 | 0.547 |
| careless_1 | human | 0.241 | -0.017 | 0.249 | 0.04 | 0.223 |

(6 of 9 raters shown.) Raw agreement badly understates the good raters because it is measured against noisy peers; the Dawid-Skene estimates land within about 0.05 of the truth and flag the careless rater (weight 0.04). The report also lists the most contested items and marks the 72 of 400 where majority vote and Dawid-Skene disagree, which is the natural queue for expert adjudication.

## Results 2: multi-agent judging, quality vs cost vs latency

### Pipeline

```
rubric.md --> [planner agent] --> criteria + judge instructions
                                      |
items.jsonl ---------------------------+--> [judge agent x criterion x tier] --> panel verdicts
                                                                                    |
                         unanimous -> accept <-- [aggregator agent] <---------------+
                         disagreement -> reconciler model call (sees every judge's rationale)
```

- **Planner** (Claude Opus 5.5) splits the rubric into independent criteria and writes self-contained judge instructions. If its criterion ids do not match the rubric's, the run falls back to the rubric text and logs why.
- **Judges**: one call per (item, criterion, model tier), run concurrently. Structured JSON output (`output_config.format`): rationale, `pass` / `fail`, confidence. Tiers: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, effort `low` by default.
- **Aggregator**: unanimous panels pass straight through, so the model is only paid for disagreements. A reconciler call re-checks the item against the criterion and weighs each judge's rationale. The panel's ratings are also written in `judgelab`'s rating format (`panel_ratings.jsonl`), so the aggregation methods and agreement statistics above run on the LLM panel directly.
- **Logging**: every call records latency, input / output tokens (thinking included), cache tokens, USD cost from the price table in `agents/pricing.py`, stop reason, request id, and the model that actually answered. Opus and Sonnet calls request server-side refusal fallbacks; if one fires, `served_model` shows it, so a fallback cannot silently mix into a per-tier comparison.

### Routing analysis

`judgelab routing` replays the call log. Each policy's verdicts come from the calls it would have made and its cost and latency are the sums of those calls, so one run of all three tiers prices every policy without extra API calls:

- single tier (haiku / sonnet / opus only)
- confidence cascades: `haiku -> opus if haiku confidence < t` (t = 0.8, 0.9, 0.95), `haiku -> sonnet`, `sonnet -> opus`
- agreement routing: `haiku + sonnet, opus only on disagreement`
- full panels: majority vote and Dawid-Skene over the three tiers
- the full pipeline: panel + aggregator agent

The report gives accuracy against ground truth with 95% Wilson intervals, total cost, cost relative to opus-only, mean latency per judgement and escalation rate, then names the cheapest policy within 1.5 pp of the most accurate one.

### Evaluation data

`judgelab make-judge-data` writes 24 customer-support refund replies (`data/support/`). Each has a store policy, an order, a customer message and the agent's reply. Defects are injected independently per criterion, so every pass / fail label is known by construction:

| Criterion | Injected defect (35% of items each) |
|---|---|
| policy_fidelity | reply misstates the return window, fee percentage or who pays shipping |
| refund_calculation | stated refund is wrong: fee or label left out, fee charged on a defective item, or a small arithmetic slip |
| no_unsupported_promises | reply offers store credit, a gift, a discount or a shipping upgrade the policy rules out |
| professional_tone | reply blames or talks down to the customer |

Some correct replies include promises the policy does allow (a prepaid label for a defective item), so a judge cannot pass the promise criterion by keyword matching alone.

### Live Claude run

Run on 2026-11-12 against the Claude API: 24 items x 4 criteria x 3 tiers, effort `low`, 8 parallel workers. 290 calls (1 planner, 288 judges, 1 reconciler), 0 errors, 0 refusal fallbacks (every call was answered by the model it asked for), 83 s wall clock, **$0.94 total** against a pre-run upper bound of $4.24. Full output: [`results/live/routing.md`](results/live/routing.md); every call, with its rationale, is in `results/live/calls.jsonl`.

| Tier | Model | Judgements | Accuracy (95% CI) | Mean latency (s) | p95 latency (s) | Cost (USD) | Cost per 100 judgements |
|---|---|---|---|---|---|---|---|
| haiku | claude-haiku-5-5 | 96 | 99.0% (94-100) | 1.63 | 2.23 | 0.0165 | 0.017 |
| sonnet | claude-sonnet-5-5 | 96 | 100.0% (96-100) | 2.05 | 2.62 | 0.2967 | 0.309 |
| opus | claude-opus-5-5 | 96 | 100.0% (96-100) | 2.24 | 2.80 | 0.5925 | 0.617 |

| Routing policy | Accuracy (95% CI) | Cost (USD) | Cost vs opus only | Mean latency per judgement (s) | Escalated |
|---|---|---|---|---|---|
| haiku only | 99.0% (94-100) | 0.0165 | 3% | 1.63 | 0% |
| haiku -> opus if haiku confidence < 0.8 | 100.0% (96-100) | 0.0226 | 4% | 1.65 | 1% |
| haiku -> opus if haiku confidence < 0.9 | 100.0% (96-100) | 0.0806 | 14% | 1.87 | 10% |
| sonnet only | 100.0% (96-100) | 0.2967 | 50% | 2.05 | 0% |
| haiku + sonnet, opus only on disagreement | 100.0% (96-100) | 0.3193 | 54% | 2.13 | 1% |
| opus only | 100.0% (96-100) | 0.5925 | 100% | 2.24 | 0% |
| pipeline: panel + aggregator agent | 100.0% (96-100) | 0.9135 | 154% | 2.36 | 1% |

(7 of 12 policies shown.) What the run shows:

- **This dataset is too easy to rank the tiers.** All three tiers score 99-100%, and the confidence intervals overlap completely. The run supports "Haiku matched Opus here at 3% of the cost". It does not support "Haiku is as good a judge as Opus". Separating the tiers needs harder items (see Limitations).
- **The one error was criterion bleed, not a reading error.** Haiku failed `policy_fidelity` on `reply_000` because the reply offered store credit. The judge instructions route offers to `no_unsupported_promises`. Haiku's own rationale says it knew this and penalised the offer anyway.
- **Confidence carried signal on that error.** Haiku gave it 0.6, its lowest confidence in the run, so the `haiku -> opus if confidence < 0.8` cascade escalated 1 judgement in 96, fixed the error, and stayed at 4% of opus-only cost. With one error this is an anecdote, not a calibration result.
- **The aggregator worked as designed.** Only 1 of 96 panels disagreed. The reconciler (Opus) ruled `pass`, citing the instruction that sends offers to another criterion, and it cost $0.008.
- **Latency was flat across tiers at effort `low`.** Mean 1.6 s for Haiku vs 2.2 s for Opus per call. Cost differed about 36x between Haiku and Opus, latency only 1.4x.

`results/simulated/` holds the same pipeline run against `SimulatedBackend`, an offline test double whose verdicts are the known truth flipped at fixed per-tier error rates. That run checks the plumbing and the routing maths; its numbers are not a measurement of any model, so they are not reported here.

## Limitations

- **Synthetic data.** Both datasets are generated. Real raters have item-dependent difficulty, drift over time and correlated errors that a fixed confusion matrix does not capture. The benchmark shows which method wins under which rater model. It does not show that real raters follow any of these models.
- **Dawid-Skene assumes conditional independence.** The `correlated_llm` scenario shows it failing: judges that share errors are over-trusted. Mitigations not implemented here: model judges from one family as one rater, or use a model with rater correlations.
- **Data hunger.** A full confusion matrix has K x (K-1) free parameters per rater. With few ratings per rater, the smoothing prior dominates; a one-coin (single accuracy) variant would be more stable and is not included.
- **No item difficulty.** Models such as GLAD, which give each item its own difficulty, are out of scope.
- **Small, easy judge benchmark.** 24 items x 4 criteria = 96 judgements per tier, so the 95% interval on an accuracy near 90% spans about 12 pp, and differences between tiers smaller than that are not significant. The live run also hit the ceiling: the templated replies, with one injected defect type per criterion, were easy for every tier (99-100%). The routing replay is only informative on data where the cheap tier makes real errors. That means longer conversations, defects that need multi-step arithmetic, or criteria that interact. This repo does not include such data yet.
- **Self-reported confidence.** The cascades route on the judge's own confidence, which LLMs calibrate poorly; the replay measures how well that works rather than assuming it.
- **Prices** are first-party list prices as of 2026-11-03 (`agents/pricing.py`); the relative comparison between tiers matters more than the dollar amounts. Prompts here are too short for prompt caching to apply.

## Repository layout

```
src/judgelab/
  io.py           JSONL ratings -> indexed sparse RatingSet
  aggregate.py    majority, weighted vote, Dawid-Skene EM, Bayes posterior
  agreement.py    Cohen, Fleiss, Krippendorff
  reliability.py  per-rater reliability, contested items, accuracy vs gold
  synth.py        rater archetypes and scenarios with known truth
  benchmark.py    multi-seed method comparison
  report.py       markdown report
  cli.py          `judgelab` command
  agents/
    backends.py   Anthropic API backend + offline simulator, per-call accounting
    pricing.py    tiers and prices
    roles.py      planner, judge, aggregator prompts and calls
    pipeline.py   orchestration (concurrent judges, reconcile on disagreement)
    routing.py    tier table and routing-policy replay
    dataset.py    synthetic support-reply data and rubric
tests/            47 tests: reference cross-checks, EM properties, CLI, pipeline with a fake API client
results/          benchmark, demo report, simulated and live pipeline runs
```

## License

MIT
