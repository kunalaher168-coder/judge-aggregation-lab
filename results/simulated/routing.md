SIMULATED backend (test double): 24 items x 4 criteria. Verdicts are the known truth flipped at fixed per-tier error rates; latency and token counts are drawn from assumed per-tier profiles; dollar figures apply real list prices to those simulated tokens. This checks the pipeline and the routing maths, it is not a measurement of any model.

| Tier | Model | Judgements | Accuracy (95% CI) | Errors | Mean latency (s) | p95 latency (s) | Cost (USD) | Cost per 100 judgements |
|---|---|---|---|---|---|---|---|---|
| haiku | claude-haiku-5-5 | 96 | 82.3% (73-89) | 0 | 1.47 | 2.59 | 0.0169 | 0.0176 |
| sonnet | claude-sonnet-5-5 | 96 | 89.6% (82-94) | 0 | 3.26 | 5.14 | 0.4912 | 0.5116 |
| opus | claude-opus-5-5 | 96 | 92.7% (86-96) | 0 | 5.32 | 7.84 | 1.1755 | 1.2245 |

| Tier | no_unsupported_promises | policy_fidelity | professional_tone | refund_calculation |
|---|---|---|---|---|
| haiku | 88% | 75% | 79% | 88% |
| sonnet | 92% | 92% | 88% | 88% |
| opus | 88% | 100% | 88% | 96% |

| Routing policy | Accuracy (95% CI) | Cost (USD) | Cost vs opus only | Mean latency per judgement (s) | Escalated |
|---|---|---|---|---|---|
| haiku only | 82.3% (73-89) | 0.0169 | 1% | 1.47 | 0% |
| haiku -> sonnet if haiku confidence < 0.9 | 89.6% (82-94) | 0.3913 | 33% | 3.96 | 76% |
| haiku -> opus if haiku confidence < 0.8 | 88.5% (81-93) | 0.4665 | 40% | 3.54 | 39% |
| sonnet only | 89.6% (82-94) | 0.4912 | 42% | 3.26 | 0% |
| haiku + sonnet, opus only on disagreement | 96.9% (91-99) | 0.8020 | 68% | 4.57 | 26% |
| haiku -> opus if haiku confidence < 0.9 | 89.6% (82-94) | 0.8993 | 77% | 5.63 | 76% |
| haiku -> opus if haiku confidence < 0.95 | 92.7% (86-96) | 1.0692 | 91% | 6.29 | 90% |
| opus only | 92.7% (86-96) | 1.1755 | 100% | 5.32 | 0% |
| sonnet -> opus if sonnet confidence < 0.9 | 93.8% (87-97) | 1.4047 | 119% | 7.45 | 77% |
| panel majority vote (all tiers) | 96.9% (91-99) | 1.6836 | 143% | 5.46 | 0% |
| panel Dawid-Skene (all tiers) | 96.9% (91-99) | 1.6836 | 143% | 5.46 | 0% |
| pipeline: panel + aggregator agent | 96.9% (91-99) | 2.0673 | 176% | 7.34 | 32% |

Cheapest policy within 1.5 pp of the most accurate: **haiku + sonnet, opus only on disagreement** (96.9% at $0.8020).
