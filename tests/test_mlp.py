import numpy as np
from rice_ml.supervised_ml import MLP


def test_linearly_separable(linearly_separable):
    X, y = linearly_separable
    clf = MLP(hidden=16, lr=0.1, epochs=30, batch=32).fit(X, y)
    assert clf.score(X, y) > 0.9


def test_xor_learnable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    X_train = np.tile(X, (50, 1)) + np.random.default_rng(0).normal(0, 0.05, size=(200, 2))
    y_train = np.tile(y, 50)
    clf = MLP(hidden=16, lr=0.1, epochs=200, batch=16).fit(X_train, y_train)
    assert clf.score(X, y) == 1.0


def test_loss_decreases(linearly_separable):
    X, y = linearly_separable
    clf = MLP(hidden=8, lr=0.05, epochs=20, batch=32).fit(X, y)
    assert clf.loss_[-1] < clf.loss_[0]


def test_probability_bounds(linearly_separable):
    X, y = linearly_separable
    clf = MLP(hidden=8, epochs=10).fit(X, y)
    p = clf.predict_proba(X)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_weight_shapes(linearly_separable):
    X, y = linearly_separable
    clf = MLP(hidden=8, epochs=2).fit(X, y)
    assert clf.W1.shape == (2, 8)
    assert clf.W2.shape == (8, 1)
