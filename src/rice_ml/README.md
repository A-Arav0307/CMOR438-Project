# rice_ml

A small from-scratch ML package for CMOR 438. Pure numpy under the hood — sklearn shows up in the notebooks for comparison, never inside the algorithms themselves.

Every model in this package follows the same `fit` / `predict` / `score` interface so the notebooks can swap them in and out without changing the surrounding code:

```python
from rice_ml.supervised_ml import LogisticRegression

clf = LogisticRegression(lr=0.1, epochs=500).fit(X_train, y_train)
preds = clf.predict(X_test)
acc = clf.score(X_test, y_test)
```

## What's where

### `supervised_ml/`

| Module | Class | What it is |
|---|---|---|
| `linear_regression.py` | `LinearRegression` | Closed-form normal equation, or batch GD via `method="gd"` |
| `logistic_regression.py` | `LogisticRegression` | Sigmoid + cross-entropy, full-batch GD |
| `perceptron.py` | `Perceptron` | Mistake-driven binary linear classifier |
| `mlp.py` | `MLP` | One hidden layer, ReLU + sigmoid output, mini-batch SGD with manual backprop |
| `knn.py` | `KNN` | Brute-force k-NN classifier on Euclidean distance |
| `decision_tree.py` | `DecisionTreeClassifier` | CART-style tree, gini or entropy impurity |
| `regression_tree.py` | `RegressionTree` | CART regressor, variance-reduction splits |
| `random_forest.py` | `RandomForestClassifier` | Bootstrap aggregation with sqrt-feature subsampling |
| `ensemble.py` | `BaggingClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `VotingClassifier` | Four ensemble methods sharing the package's trees |
| `distance_metrics.py` | — | Euclidean, Manhattan, Chebyshev, Minkowski |

### `unsupervised_ml/`

| Module | Class | What it is |
|---|---|---|
| `pca.py` | `PCA` | Eigendecomposition of the covariance matrix |
| `kmeans.py` | `KMeans` | Lloyd's algorithm with random init |
| `dbscan.py` | `DBSCAN` | Region-query expansion, no kd-tree |

### `preprocessing/`

`standardize`, `minmax_scale`, `train_test_split`. Used both inside the notebooks and the test fixtures.

### `measures/`

`accuracy_score`, `mean_squared_error`, `mean_absolute_error`, `r2_score`, `confusion_matrix`. Implementations of the basic metrics from scratch so the package doesn't depend on `sklearn.metrics` either.

## Performance notes

The trees, the ensembles built on them, and DBSCAN are not vectorized — split search is `O(d·n²)` per node and DBSCAN recomputes pairwise distances on every region query. They run fine on 1k–2k row subsamples (1–10 seconds) but trying to fit them on the full 13k-row California housing dataset will take a long time. The notebooks subsample for the from-scratch comparisons and call this out explicitly in the markdown.

Linear regression's `pinv` closed-form, the MLP's mini-batch SGD, and the unsupervised methods are vectorized and fast on the full dataset.

## Tests

```
pytest
```

See `tests/README.md` for what's tested and how the fixtures work.
