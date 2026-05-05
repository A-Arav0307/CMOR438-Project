# Regression Tree

Decision tree regressor on `median_house_value`. A tree recursively splits the feature space; each leaf predicts the mean target value of the training points that fall in it.

## How splits get chosen

At each node, we search every feature and every threshold for the split that minimizes total child variance:

$$\text{score}(j, t) = \frac{n_L}{n} \text{Var}(y_L) + \frac{n_R}{n} \text{Var}(y_R)$$

That's "squared error" criterion. The notebook also runs `friedman_mse`, `absolute_error`, and `poisson` for comparison.

## Why depth matters

A tree with no depth limit will fit the training set perfectly (a leaf per row) and generalize poorly. The depth sweep makes that visible — train R² climbs to 1.0, test R² peaks somewhere around depth 8–10 and falls off after that.

## Sections

1. Same preprocessing pipeline as earlier regression notebooks
2. Default sklearn baseline (no depth limit)
3. Depth sweep, `max_depth ∈ {2, 4, 6, 8, 10, 15, None}`, plot train and test R² together
4. Splitting criterion comparison at a fixed depth
5. Feature importances — `median_income` and the geographic features dominate, as expected
6. Predicted-vs-actual at the best depth
7. Plot the actual tree at depth 3 so the splits are readable
8. From-scratch `rice_ml.RegressionTree` against sklearn

## Notes

Trees don't need feature scaling at all — the splits are threshold-based, so z-scoring the inputs is wasted work for this notebook specifically. I left the standardization in anyway so the preprocessing matches every other notebook in this folder.
