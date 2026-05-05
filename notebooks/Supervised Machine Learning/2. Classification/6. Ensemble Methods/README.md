# Ensemble Methods

Four ensembles compared on the above-median classification task. Each one combines weak base learners (usually shallow trees) but does it differently.

## The four

| Method | Idea | Base learner here |
|---|---|---|
| **Bagging** | Average predictions over bootstrap-trained models | Decision trees |
| **AdaBoost** | Iteratively reweight misclassified points; final prediction is a weighted vote | Stumps |
| **Gradient Boosting** | Each new tree fits the residual of the running prediction | Shallow regression trees on log-odds |
| **Voting** | Average heterogeneous models (LR + DT + KNN here) | Mixed |

Bagging reduces variance. Boosting reduces bias. Voting is closer to ensembling diverse model *types* rather than diverse model *fits*.

## Sections

1. Same above-median preprocessing
2. **Bagging** — sklearn `BaggingClassifier`, `n_estimators` sweep
3. **AdaBoost** — `AdaBoostClassifier`, sweep over `n_estimators` and `learning_rate`
4. **Gradient Boosting** — `GradientBoostingClassifier`, dedicated learning rate sweep because GBM is the most sensitive of the four
5. **Voting** — hard vs soft (averaging probabilities is usually better when the base classifiers are calibrated)
6. Stacked comparison plot — accuracy of all four side by side
7. From-scratch implementations from `rice_ml.ensemble` (Bagging, AdaBoost, GBM) — same data, same hyperparameters, sanity-check accuracies

## Notes

The four methods cluster within about 2 percentage points of each other on this dataset (~83–85%). On other datasets the gap can be much larger; here the signal is mostly captured by `median_income` + the geographic features, and any reasonable ensemble extracts that signal. The GBM learning-rate sweep is the most instructive plot — it shows the classic boosting bias-variance tradeoff (small lr + many trees ≈ slow but steady, large lr + few trees ≈ noisy and underfit).
