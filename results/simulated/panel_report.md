# Judgement aggregation report: panel_ratings.jsonl

## Criterion: no_unsupported_promises

24 items, 3 raters, 72 ratings, classes ['fail', 'pass']

### Inter-rater agreement

| Statistic | Value |
|---|---|
| percent_agreement | 0.778 |
| cohen_kappa_mean | 0.554 |
| fleiss_kappa | 0.552 |
| krippendorff_alpha_nominal | 0.559 |

### Aggregation

| Method | Ties | Notes | Accuracy vs truth | MAE |
|---|---|---|---|---|
| majority | 0 |  | 1.000 | - |
| weighted | 0 |  | 1.000 | - |
| dawid_skene | 0 | EM 15 iterations, converged=True | 1.000 | - |

Majority vote and Dawid-Skene disagree on 0 of 24 items.

### Per-rater reliability

loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.

| rater | type | n_ratings | loo_agreement | mean_kappa | ds_accuracy | weight | true_accuracy |
|---|---|---|---|---|---|---|---|
| sonnet | llm | 24 | 0.792 | 0.580 | 0.866 | 1.204 | 0.917 |
| opus | llm | 24 | 0.771 | 0.542 | 0.826 | 1.099 | 0.875 |
| haiku | llm | 24 | 0.771 | 0.539 | 0.825 | 1.099 | 0.875 |

### Most contested items (top 5)

| item | ratings | top share | majority | Dawid-Skene (conf) | labels |
|---|---|---|---|---|---|
| reply_003 | 3 | 0.67 | pass | pass (0.77) | haiku=pass, opus=pass, sonnet=fail |
| reply_023 | 3 | 0.67 | fail | fail (0.80) | haiku=fail, opus=fail, sonnet=pass |
| reply_014 | 3 | 0.67 | pass | pass (0.85) | haiku=pass, opus=fail, sonnet=pass |
| reply_015 | 3 | 0.67 | pass | pass (0.85) | haiku=pass, opus=fail, sonnet=pass |
| reply_012 | 3 | 0.67 | pass | pass (0.86) | haiku=fail, opus=pass, sonnet=pass |

`*` = majority vote and Dawid-Skene pick different labels.

## Criterion: policy_fidelity

24 items, 3 raters, 72 ratings, classes ['fail', 'pass']

### Inter-rater agreement

| Statistic | Value |
|---|---|
| percent_agreement | 0.778 |
| cohen_kappa_mean | 0.552 |
| fleiss_kappa | 0.552 |
| krippendorff_alpha_nominal | 0.559 |

### Aggregation

| Method | Ties | Notes | Accuracy vs truth | MAE |
|---|---|---|---|---|
| majority | 0 |  | 1.000 | - |
| weighted | 0 |  | 1.000 | - |
| dawid_skene | 0 | EM 13 iterations, converged=True | 1.000 | - |

Majority vote and Dawid-Skene disagree on 0 of 24 items.

### Per-rater reliability

loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.

| rater | type | n_ratings | loo_agreement | mean_kappa | ds_accuracy | weight | true_accuracy |
|---|---|---|---|---|---|---|---|
| opus | llm | 24 | 0.833 | 0.664 | 0.943 | 2.037 | 1.000 |
| sonnet | llm | 24 | 0.792 | 0.580 | 0.891 | 2.037 | 0.917 |
| haiku | llm | 24 | 0.708 | 0.413 | 0.721 | 0.811 | 0.750 |

### Most contested items (top 5)

| item | ratings | top share | majority | Dawid-Skene (conf) | labels |
|---|---|---|---|---|---|
| reply_022 | 3 | 0.67 | pass | pass (0.83) | haiku=pass, opus=pass, sonnet=fail |
| reply_008 | 3 | 0.67 | fail | fail (0.84) | haiku=fail, opus=fail, sonnet=pass |
| reply_004 | 3 | 0.67 | pass | pass (0.98) | haiku=fail, opus=pass, sonnet=pass |
| reply_017 | 3 | 0.67 | pass | pass (0.98) | haiku=fail, opus=pass, sonnet=pass |
| reply_020 | 3 | 0.67 | pass | pass (0.98) | haiku=fail, opus=pass, sonnet=pass |

`*` = majority vote and Dawid-Skene pick different labels.

## Criterion: professional_tone

24 items, 3 raters, 72 ratings, classes ['fail', 'pass']

### Inter-rater agreement

| Statistic | Value |
|---|---|
| percent_agreement | 0.778 |
| cohen_kappa_mean | 0.522 |
| fleiss_kappa | 0.518 |
| krippendorff_alpha_nominal | 0.525 |

### Aggregation

| Method | Ties | Notes | Accuracy vs truth | MAE |
|---|---|---|---|---|
| majority | 0 |  | 0.875 | - |
| weighted | 0 |  | 0.875 | - |
| dawid_skene | 0 | EM 20 iterations, converged=True | 0.875 | - |

Majority vote and Dawid-Skene disagree on 0 of 24 items.

### Per-rater reliability

loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.

| rater | type | n_ratings | loo_agreement | mean_kappa | ds_accuracy | weight | true_accuracy |
|---|---|---|---|---|---|---|---|
| sonnet | llm | 24 | 0.792 | 0.548 | 0.869 | 1.435 | 0.875 |
| opus | llm | 24 | 0.792 | 0.548 | 0.869 | 1.435 | 0.875 |
| haiku | llm | 24 | 0.750 | 0.471 | 0.783 | 0.999 | 0.792 |

### Most contested items (top 5)

| item | ratings | top share | majority | Dawid-Skene (conf) | labels |
|---|---|---|---|---|---|
| reply_017 | 3 | 0.67 | fail | fail (0.75) | haiku=fail, opus=fail, sonnet=pass |
| reply_008 | 3 | 0.67 | fail | fail (0.75) | haiku=fail, opus=pass, sonnet=fail |
| reply_003 | 3 | 0.67 | pass | pass (0.82) | haiku=pass, opus=fail, sonnet=pass |
| reply_014 | 3 | 0.67 | pass | pass (0.82) | haiku=pass, opus=pass, sonnet=fail |
| reply_010 | 3 | 0.67 | fail | fail (0.89) | haiku=pass, opus=fail, sonnet=fail |

`*` = majority vote and Dawid-Skene pick different labels.

## Criterion: refund_calculation

24 items, 3 raters, 72 ratings, classes ['fail', 'pass']

### Inter-rater agreement

| Statistic | Value |
|---|---|
| percent_agreement | 0.806 |
| cohen_kappa_mean | 0.611 |
| fleiss_kappa | 0.611 |
| krippendorff_alpha_nominal | 0.616 |

### Aggregation

| Method | Ties | Notes | Accuracy vs truth | MAE |
|---|---|---|---|---|
| majority | 0 |  | 1.000 | - |
| weighted | 0 |  | 1.000 | - |
| dawid_skene | 0 | EM 18 iterations, converged=True | 1.000 | - |

Majority vote and Dawid-Skene disagree on 0 of 24 items.

### Per-rater reliability

loo_agreement = agreement with the majority of the other raters; ds_accuracy = accuracy implied by the rater's Dawid-Skene confusion matrix; weight = weighted-vote weight.

| rater | type | n_ratings | loo_agreement | mean_kappa | ds_accuracy | weight | true_accuracy |
|---|---|---|---|---|---|---|---|
| opus | llm | 24 | 0.833 | 0.669 | 0.916 | 1.435 | 0.958 |
| sonnet | llm | 24 | 0.792 | 0.583 | 0.831 | 1.204 | 0.875 |
| haiku | llm | 24 | 0.792 | 0.583 | 0.831 | 1.204 | 0.875 |

### Most contested items (top 5)

| item | ratings | top share | majority | Dawid-Skene (conf) | labels |
|---|---|---|---|---|---|
| reply_020 | 3 | 0.67 | fail | fail (0.66) | haiku=fail, opus=pass, sonnet=fail |
| reply_000 | 3 | 0.67 | fail | fail (0.91) | haiku=pass, opus=fail, sonnet=fail |
| reply_018 | 3 | 0.67 | fail | fail (0.91) | haiku=fail, opus=fail, sonnet=pass |
| reply_011 | 3 | 0.67 | pass | pass (0.93) | haiku=fail, opus=pass, sonnet=pass |
| reply_014 | 3 | 0.67 | pass | pass (0.93) | haiku=fail, opus=pass, sonnet=pass |

`*` = majority vote and Dawid-Skene pick different labels.
