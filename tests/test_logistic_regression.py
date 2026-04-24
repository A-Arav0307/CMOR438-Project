import numpy as np
from rice_ml.supervised_ml import LogisticRegression
from rice_ml.supervised_ml.logistic_regression import sigmoid


def test_sigmoid_bounds():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    s = sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert abs(s[2] - 0.5) < 1e-9


def test_linearly_separable_high_acc(linearly_separable):
    X, y = linearly_separable
    clf = LogisticRegression(lr=0.1, epochs=500).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_loss_decreases(linearly_separable):
    X, y = linearly_separable
    clf = LogisticRegression(lr=0.1, epochs=100).fit(X, y)
    assert clf.loss_[-1] < clf.loss_[0]


def test_probabilities_in_range(linearly_separable):
    X, y = linearly_separable
    clf = LogisticRegression(epochs=50).fit(X, y)
    p = clf.predict_proba(X)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_threshold_affects_predictions(linearly_separable):
    X, y = linearly_separable
    clf = LogisticRegression(epochs=50).fit(X, y)
    low = clf.predict(X, threshold=0.1).sum()
    high = clf.predict(X, threshold=0.9).sum()
    assert low >= high
