import numpy as np
from rice_ml.supervised_ml import Perceptron


def test_linearly_separable_converges(linearly_separable):
    X, y = linearly_separable
    clf = Perceptron(eta=0.1, epochs=30).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    clf = Perceptron(epochs=5).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_reproducibility(linearly_separable):
    X, y = linearly_separable
    a = Perceptron(seed=7, epochs=10).fit(X, y).predict(X)
    b = Perceptron(seed=7, epochs=10).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_weights_exist_after_fit(linearly_separable):
    X, y = linearly_separable
    clf = Perceptron(epochs=3).fit(X, y)
    assert clf.w.shape == (X.shape[1],)
    assert isinstance(clf.b, float)
    assert len(clf.errors_) == 3
