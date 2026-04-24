import numpy as np
import pytest


@pytest.fixture
def linearly_separable():
    rng = np.random.default_rng(0)
    n = 200
    X0 = rng.normal(loc=-2.0, scale=0.5, size=(n, 2))
    X1 = rng.normal(loc=2.0, scale=0.5, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    centers = np.array([[-5, -5], [5, 5], [-5, 5]], dtype=float)
    pts = []
    for c in centers:
        pts.append(rng.normal(loc=c, scale=0.6, size=(100, 2)))
    X = np.vstack(pts)
    return X


@pytest.fixture
def correlated_2d():
    rng = np.random.default_rng(2)
    t = rng.normal(0, 1, size=500)
    X = np.column_stack([t, 2 * t + rng.normal(0, 0.05, size=500)])
    return X
