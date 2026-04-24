import numpy as np


class KMeans:
    def __init__(self, k=5, max_iter=100, tol=1e-6, seed=42):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), self.k, replace=False)
        self.centroids_ = X[idx].copy()
        self.inertia_history_ = []
        labels = np.zeros(len(X), dtype=int)
        for _ in range(self.max_iter):
            d = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)
            labels = d.argmin(axis=1)
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if (labels == c).any() else self.centroids_[c]
                for c in range(self.k)
            ])
            inertia = float(np.sum(np.min(d, axis=1) ** 2))
            self.inertia_history_.append(inertia)
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            if shift < self.tol:
                break
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)
        return d.argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
