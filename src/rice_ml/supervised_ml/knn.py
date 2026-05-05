"""Brute-force k-nearest-neighbors classifier."""
import numpy as np


class KNN:
    """k-NN classifier using Euclidean distance and majority vote.

    Parameters:
        k : number of neighbors to consult per prediction
    """

    def __init__(self, k=5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.empty(len(X), dtype=self.y.dtype)
        for i, x in enumerate(X):
            d = np.linalg.norm(self.X - x, axis=1)
            idx = np.argpartition(d, min(self.k, len(d) - 1))[:self.k]
            vals, counts = np.unique(self.y[idx], return_counts=True)
            preds[i] = vals[counts.argmax()]
        return preds

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
