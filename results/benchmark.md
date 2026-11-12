Synthetic benchmark: 400 items x 20 seeds per scenario, 4-point rubric. Accuracy vs ground truth, mean (sd).

| Scenario | Majority | Weighted | Dawid-Skene | Oracle (true confusions) | DS - MV (pp) | DS wins / losses |
|---|---|---|---|---|---|---|
| independent | 86.5 (1.7) | 87.7 (1.6) | 87.8 (1.9) | 89.6 (1.6) | +1.3 | 15 / 4 |
| spammers | 68.2 (1.9) | 79.8 (2.0) | 76.8 (3.3) | 81.8 (2.1) | +8.7 | 20 / 0 |
| biased_llm | 78.2 (1.8) | 82.9 (2.2) | 90.5 (2.0) | 92.2 (1.4) | +12.2 | 20 / 0 |
| mixed | 74.0 (2.6) | 80.5 (2.4) | 86.7 (2.3) | 89.6 (1.6) | +12.7 | 20 / 0 |
| correlated_llm | 71.9 (2.6) | 69.1 (2.9) | 66.2 (2.8) | 76.9 (2.4) | -5.7 | 0 / 20 |

| Scenario | MAE majority | MAE Dawid-Skene | Rater-accuracy error, DS (abs) | Rater-accuracy error, leave-one-out (abs) | Spearman(DS est, true) |
|---|---|---|---|---|---|
| independent | 0.142 | 0.126 | 0.028 | 0.085 | 0.88 |
| spammers | 0.406 | 0.252 | 0.034 | 0.090 | 0.90 |
| biased_llm | 0.221 | 0.096 | 0.030 | 0.099 | 0.94 |
| mixed | 0.271 | 0.136 | 0.034 | 0.097 | 0.92 |
| correlated_llm | 0.311 | 0.371 | 0.204 | 0.163 | -0.70 |
