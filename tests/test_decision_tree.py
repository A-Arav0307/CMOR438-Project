import numpy as np
import pytest
from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier


def test_separable_perfect_fit(linearly_separable):
    X, y = linearly_separable
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.score(X, y) > 0.95


def test_max_depth_one_underfits_xor():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.uniform(-1, 1, size=(4 * n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    shallow = DecisionTreeClassifier(max_depth=1).fit(X, y).score(X, y)
    deep = DecisionTreeClassifier(max_depth=5).fit(X, y).score(X, y)
    assert deep > shallow


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    preds = DecisionTreeClassifier(max_depth=3).fit(X, y).predict(X)
    assert preds.shape == (len(X),)


def test_entropy_works(linearly_separable):
    X, y = linearly_separable
    clf = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    assert clf.score(X, y) > 0.9


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bogus")
