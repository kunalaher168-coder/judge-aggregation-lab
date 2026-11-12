Backend: anthropic. 24 items x 4 criteria. Wall clock 83.4s with 8 parallel workers; total spend $0.9362 (planner + judges + aggregator).

| Tier | Model | Judgements | Accuracy (95% CI) | Errors | Mean latency (s) | p95 latency (s) | Cost (USD) | Cost per 100 judgements |
|---|---|---|---|---|---|---|---|---|
| haiku | claude-haiku-5-5 | 96 | 99.0% (94-100) | 0 | 1.63 | 2.23 | 0.0165 | 0.0171 |
| sonnet | claude-sonnet-5-5 | 96 | 100.0% (96-100) | 0 | 2.05 | 2.62 | 0.2967 | 0.3091 |
| opus | claude-opus-5-5 | 96 | 100.0% (96-100) | 0 | 2.24 | 2.80 | 0.5925 | 0.6171 |

| Tier | no_unsupported_promises | policy_fidelity | professional_tone | refund_calculation |
|---|---|---|---|---|
| haiku | 100% | 96% | 100% | 100% |
| sonnet | 100% | 100% | 100% | 100% |
| opus | 100% | 100% | 100% | 100% |

| Routing policy | Accuracy (95% CI) | Cost (USD) | Cost vs opus only | Mean latency per judgement (s) | Escalated |
|---|---|---|---|---|---|
| haiku only | 99.0% (94-100) | 0.0165 | 3% | 1.63 | 0% |
| haiku -> opus if haiku confidence < 0.8 | 100.0% (96-100) | 0.0226 | 4% | 1.65 | 1% |
| haiku -> sonnet if haiku confidence < 0.9 | 100.0% (96-100) | 0.0484 | 8% | 1.85 | 10% |
| haiku -> opus if haiku confidence < 0.9 | 100.0% (96-100) | 0.0806 | 14% | 1.87 | 10% |
| sonnet only | 100.0% (96-100) | 0.2967 | 50% | 2.05 | 0% |
| sonnet -> opus if sonnet confidence < 0.9 | 100.0% (96-100) | 0.2967 | 50% | 2.05 | 0% |
| haiku + sonnet, opus only on disagreement | 100.0% (96-100) | 0.3193 | 54% | 2.13 | 1% |
| haiku -> opus if haiku confidence < 0.95 | 100.0% (96-100) | 0.3199 | 54% | 2.77 | 51% |
| opus only | 100.0% (96-100) | 0.5925 | 100% | 2.24 | 0% |
| panel majority vote (all tiers) | 100.0% (96-100) | 0.9056 | 153% | 2.33 | 0% |
| panel Dawid-Skene (all tiers) | 100.0% (96-100) | 0.9056 | 153% | 2.33 | 0% |
| pipeline: panel + aggregator agent | 100.0% (96-100) | 0.9135 | 154% | 2.36 | 1% |

Cheapest policy within 1.5 pp of the most accurate: **haiku only** (99.0% at $0.0165).
