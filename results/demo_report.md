# Judgement aggregation report: ratings.jsonl

## Criterion: overall

400 items, 9 raters, 2000 ratings, classes [1, 2, 3, 4]

### Inter-rater agreement

| Statistic | Value |
|---|---|
| percent_agreement | 0.408 |
| cohen_kappa_mean | 0.192 |
| fleiss_kappa | 0.188 |
| krippendorff_alpha_nominal | 0.189 |
| cohen_kappa_quadratic_mean | 0.417 |
| krippendorff_alpha_ordinal | 0.400 |
| krippendorff_alpha_interval | 0.403 |

### Aggregation

| Method | Ties | Notes | Accuracy vs truth | MAE |
|---|---|---|---|---|
| majority | 81 |  | 0.787 | 0.225 |
| weighted | 0 |  | 0.830 | 0.175 |
| dawid_skene | 0 | EM 34 iterations, converged=True | 0.875 | 0.133 |

Majority vote and Dawid-Skene disagree on 72 of 400 items.

### Per-rater reliability

loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.

| rater | type | n_ratings | loo_agreement | mean_kappa | ds_accuracy | weight | true_accuracy |
|---|---|---|---|---|---|---|---|
| human_1 | human | 217 | 0.596 | 0.283 | 0.810 | 1.631 | 0.834 |
| llm_strong | llm | 223 | 0.563 | 0.302 | 0.757 | 1.617 | 0.807 |
| human_2 | human | 228 | 0.491 | 0.215 | 0.622 | 1.186 | 0.645 |
| human_3 | human | 229 | 0.481 | 0.197 | 0.582 | 1.142 | 0.616 |
| llm_lenient_b | llm | 230 | 0.452 | 0.200 | 0.554 | 1.030 | 0.552 |
| llm_weak | llm | 217 | 0.490 | 0.202 | 0.527 | 1.108 | 0.525 |
| llm_lenient_a | llm | 229 | 0.430 | 0.173 | 0.510 | 1.003 | 0.507 |
| llm_harsh | llm | 212 | 0.413 | 0.169 | 0.506 | 0.873 | 0.547 |
| careless_1 | human | 215 | 0.241 | -0.017 | 0.249 | 0.043 | 0.223 |

### Most contested items (top 10)

| item | ratings | top share | majority | Dawid-Skene (conf) | labels |
|---|---|---|---|---|---|
| item_118 * | 5 | 0.40 | 2 | 1 (0.49) | human_1=1, human_2=2, human_3=3, llm_lenient_b=4, llm_weak=2 |
| item_121 * | 5 | 0.40 | 1 | 2 (0.54) | careless_1=1, human_3=3, llm_lenient_b=2, llm_strong=4, llm_weak=1 |
| item_377 | 5 | 0.40 | 3 | 3 (0.70) | careless_1=1, human_1=3, human_2=2, llm_lenient_b=2, llm_weak=3 |
| item_183 * | 5 | 0.40 | 4 | 2 (0.76) | careless_1=4, human_1=3, llm_harsh=4, llm_lenient_b=2, llm_strong=2 |
| item_164 | 5 | 0.40 | 3 | 3 (0.76) | careless_1=1, human_2=3, llm_lenient_a=2, llm_lenient_b=3, llm_strong=4 |
| item_230 * | 5 | 0.40 | 3 | 1 (0.73) | careless_1=3, human_1=3, human_3=1, llm_lenient_a=2, llm_lenient_b=4 |
| item_277 * | 5 | 0.40 | 3 | 2 (0.66) | human_1=3, human_3=3, llm_lenient_b=2, llm_strong=1, llm_weak=2 |
| item_092 | 5 | 0.40 | 1 | 1 (0.50) | careless_1=4, human_2=1, human_3=2, llm_strong=1, llm_weak=3 |
| item_367 | 5 | 0.40 | 2 | 2 (0.50) | careless_1=4, human_1=1, human_3=2, llm_lenient_a=2, llm_weak=3 |
| item_191 | 5 | 0.40 | 3 | 3 (0.51) | human_1=2, llm_harsh=3, llm_lenient_a=4, llm_strong=3, llm_weak=2 |

`*` = majority vote and Dawid-Skene pick different labels.
