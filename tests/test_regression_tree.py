import numpy as np
from rice_ml.supervised_ml.regression_tree import RegressionTree


def test_step_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(200, 1))
    y = np.where(X[:, 0] < 5, 0.0, 10.0) + rng.normal(0, 0.5, size=200)
    tree = RegressionTree(max_depth=3).fit(X, y)
    assert tree.score(X, y) > 0.8


def test_zero_depth_returns_mean():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)
    tree = RegressionTree(max_depth=0).fit(X, y)
    pred = tree.predict(X)
    assert np.allclose(pred, y.mean())


def test_deeper_fits_better():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = np.sin(3 * X[:, 0]) + np.cos(3 * X[:, 1])
    shallow = RegressionTree(max_depth=2).fit(X, y).score(X, y)
    deep = RegressionTree(max_depth=8).fit(X, y).score(X, y)
    assert deep > shallow


def test_predict_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    pred = RegressionTree(max_depth=4).fit(X, y).predict(X)
    assert pred.shape == (40,)
