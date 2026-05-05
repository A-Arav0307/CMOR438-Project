import numpy as np
from rice_ml.preprocessing import standardize, minmax_scale, train_test_split


def test_standardize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(100, 4))
    Z = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-10)


def test_standardize_constant_column():
    X = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
    Z = standardize(X)
    assert np.allclose(Z[:, 0], 0)


def test_minmax_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    Z = minmax_scale(X)
    assert np.allclose(Z.min(axis=0), 0)
    assert np.allclose(Z.max(axis=0), 1)


def test_train_test_split_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)
    assert len(Xtr) == 40
    assert len(Xte) == 10
    assert len(ytr) == 40
    assert len(yte) == 10


def test_train_test_split_no_overlap():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=0)
    train_set = set(ytr.tolist())
    test_set = set(yte.tolist())
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(range(100))


def test_train_test_split_seeded_differs():
    X = np.arange(100).reshape(100, 1)
    a = train_test_split(X, test_size=0.2, seed=0)[1]
    b = train_test_split(X, test_size=0.2, seed=1)[1]
    assert not np.array_equal(a, b)
