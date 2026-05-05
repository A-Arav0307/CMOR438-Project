# Random Forest

A random forest is an average over many decision trees, where each tree sees:

1. A bootstrap sample of the training rows (sample with replacement, same size as the original)
2. A random subset of `sqrt(n_features)` features at each split

That double randomization is what makes the trees disagree — and averaging over disagreeing models is what kills the variance problem from the single-tree notebook.

## OOB score (free validation)

Each bootstrap sample leaves out roughly $1 - (1 - 1/n)^n \approx 36.8\%$ of the rows. Those out-of-bag points can score the tree without touching the test set. Averaging OOB predictions across the forest gives a free validation estimate that closely tracks held-out test accuracy.

## Sections

1. Same above-median preprocessing
2. sklearn `RandomForestClassifier` baseline
3. `n_estimators` sweep — accuracy keeps climbing until it plateaus around 100 trees, then flatlines
4. `max_depth` sweep at fixed `n_estimators`
5. Feature importances averaged across trees (more stable than a single tree's importances)
6. OOB score logged alongside test accuracy at each `n_estimators`
7. ROC curve
8. From-scratch `rice_ml.RandomForestClassifier` on a smaller subsample for sanity check

## Notes

The forest lands around **84%** test accuracy, very close to the MLP. The interesting part isn't the headline number — it's how stable the importances become once you're averaging over 100 trees. A single tree's importances flip wildly under reseeding; the forest's don't.
