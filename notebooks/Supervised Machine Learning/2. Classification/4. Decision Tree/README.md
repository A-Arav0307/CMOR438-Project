# Decision Tree

Classification version of the regression-tree notebook. Same recursive splitting idea, but each leaf now predicts a class label (or class probability) instead of a mean.

## How splits get chosen

For each candidate split, score the two children by impurity and weight by sample count:

$$\text{score}(j, t) = \frac{n_L}{n} I(y_L) + \frac{n_R}{n} I(y_R)$$

Two impurity functions are compared:

- **Gini**: $I = 1 - \sum_k p_k^2$
- **Entropy**: $I = -\sum_k p_k \log p_k$
- **log_loss**: same as entropy in scikit-learn (alias)

In practice gini and entropy give nearly identical trees — the splits they prefer agree most of the time.

## Why the depth sweep matters

With no depth limit, a CART tree drives training accuracy to 100% by carving leaves that contain a single point. Test accuracy collapses. The depth sweep makes the overfitting curve visible: train accuracy keeps climbing while test peaks around depth 6–8, then falls off.

## Sections

1. Same above-median preprocessing
2. Default baseline (unlimited depth)
3. Depth sweep with both train and test accuracy plotted
4. Criterion comparison (gini / entropy / log_loss)
5. Feature importances
6. ROC curve at the best-depth tree
7. Plot the depth-3 tree with sklearn's `plot_tree` so the splits are actually readable
8. From-scratch tree from `rice_ml.DecisionTreeClassifier` on a 2,000-row subsample

## Notes

A single tree is interpretable but high-variance — small data perturbations produce very different trees. That motivates the next two notebooks (Random Forest and the ensembles), which average over many trees to reduce variance.
