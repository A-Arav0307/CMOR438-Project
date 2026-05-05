# Tests

Pytest suite for the `rice_ml` package — 75 tests across 15 files, all deterministic (seeded numpy RNGs throughout).

## Running

```
pip install -e ".[test]"
pytest                          # everything
pytest tests/test_mlp.py        # one file
pytest -k perceptron            # by name pattern
pytest -q                       # quiet mode
```

The pytest config lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`pythonpath = ["src"]` means tests work without an editable install, but `pip install -e .` is still recommended so notebooks pick up the package too.

## Layout

One test file per module:

```
tests/
  conftest.py                       shared fixtures
  test_perceptron.py
  test_logistic_regression.py
  test_mlp.py
  test_knn.py
  test_kmeans.py
  test_pca.py
  test_dbscan.py
  test_linear_regression.py
  test_decision_tree.py
  test_regression_tree.py
  test_random_forest.py
  test_ensemble.py
  test_distance_metrics.py
  test_preprocessing.py
  test_measures.py
```

## What each file checks

For supervised models:
- Convergence on a known-easy problem (linearly separable blobs, etc.) — `score(X, y) > 0.95` or similar
- Output shape — `predict(X)` returns the right length
- Reproducibility — same `seed` produces the same predictions
- At least one `ValueError` test for invalid arguments

For unsupervised models:
- Recovers planted structure (3 blobs ⇒ 3 clusters for K-Means / DBSCAN)
- Output shape and label consistency
- Specific properties (PCA's variance ratio sums to 1, etc.)

For utilities (preprocessing, measures, distance metrics):
- The math is correct on hand-computable inputs (e.g., `euclidean([0,0], [3,4]) == 5.0`)
- Edge cases that matter (constant column in `standardize`, length mismatch in `accuracy_score`)

## Fixtures (`conftest.py`)

Three seeded fixtures used across multiple test files:

- `linearly_separable` — two Gaussian blobs at ±2, 400 points, binary labels. Used by all supervised classifier tests.
- `blobs` — three Gaussians at distinct centers. Used by clustering tests.
- `correlated_2d` — collinear-ish 2D data. Used by PCA tests.

Putting them in `conftest.py` keeps each test file short and means the same data goes into every model that needs a "linearly separable" sanity check.
