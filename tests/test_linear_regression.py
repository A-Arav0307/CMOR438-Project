import numpy as np
import pytest
from rice_ml.supervised_ml.linear_regression import LinearRegression


def test_normal_eq_recovers_coef():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_coef = np.array([1.5, -2.0, 0.5])
    true_intercept = 3.0
    y = X @ true_coef + true_intercept + rng.normal(0, 0.1, size=200)

    lr = LinearRegression(method="normal").fit(X, y)
    assert np.allclose(lr.coef_, true_coef, atol=0.1)
    assert abs(lr.intercept_ - true_intercept) < 0.1


def test_gd_converges_to_normal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.05, size=200)

    a = LinearRegression(method="normal").fit(X, y)
    b = LinearRegression(method="gd", lr=0.05, epochs=2000).fit(X, y)
    assert np.allclose(a.coef_, b.coef_, atol=0.1)


def test_predict_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50)
    preds = LinearRegression().fit(X, y).predict(X)
    assert preds.shape == (50,)


def test_score_perfect_fit():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 2))
    y = X @ np.array([2.0, -1.0]) + 1.0
    lr = LinearRegression().fit(X, y)
    assert lr.score(X, y) > 0.999


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        LinearRegression(method="bogus")
