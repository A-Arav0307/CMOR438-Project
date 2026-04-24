import numpy as np


class PCA:
    def __init__(self, n_components):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        self.components_ = vecs[:, order[:self.n_components]].T
        self.explained_variance_ = vals[order][:self.n_components]
        total = vals.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(self.n_components)
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_
