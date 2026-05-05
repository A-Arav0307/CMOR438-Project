import numpy as np


def standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma


def minmax_scale(X):
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / rng


def train_test_split(X, y=None, test_size=0.25, shuffle=True, seed=42):
    X = np.asarray(X)
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    n_test = int(np.round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    if y is None:
        return X[train_idx], X[test_idx]
    y = np.asarray(y)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


__all__ = ["standardize", "minmax_scale", "train_test_split"]
