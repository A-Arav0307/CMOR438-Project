import numpy as np
import pytest
from rice_ml.supervised_ml import KNN


def test_invalid_k():
    with pytest.raises(ValueError):
        KNN(k=0)


def test_linearly_separable(linearly_separable):
    X, y = linearly_separable
    clf = KNN(k=5).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_k_equals_one_memorizes_training(linearly_separable):
    X, y = linearly_separable
    clf = KNN(k=1).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    clf = KNN(k=3).fit(X, y)
    preds = clf.predict(X[:10])
    assert preds.shape == (10,)
