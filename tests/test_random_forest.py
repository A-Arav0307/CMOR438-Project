import numpy as np
from rice_ml.supervised_ml.random_forest import RandomForestClassifier


def test_separable(linearly_separable):
    X, y = linearly_separable
    rf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X, y)
    assert rf.score(X, y) > 0.9


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    preds = RandomForestClassifier(n_estimators=5).fit(X, y).predict(X)
    assert preds.shape == (len(X),)


def test_reproducibility(linearly_separable):
    X, y = linearly_separable
    a = RandomForestClassifier(n_estimators=5, seed=7).fit(X, y).predict(X)
    b = RandomForestClassifier(n_estimators=5, seed=7).fit(X, y).predict(X)
    assert np.array_equal(a, b)
