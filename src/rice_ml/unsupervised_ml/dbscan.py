"""DBSCAN — density-based clustering with brute-force region queries."""
import numpy as np


class DBSCAN:
    """Density-based spatial clustering. Labels each point as a cluster id or -1 (noise).

    Parameters:
        eps : neighborhood radius
        min_samples : minimum neighbors (including self) for a point to count as core
    """

    def __init__(self, eps=0.5, min_samples=5):
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster
            seeds = list(neighbors)
            in_seeds = set(seeds)
            k = 0
            while k < len(seeds):
                j = seeds[k]
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = self._region(X, j)
                    if len(j_neighbors) >= self.min_samples:
                        for nb in j_neighbors:
                            if nb not in in_seeds:
                                seeds.append(nb)
                                in_seeds.add(nb)
                if labels[j] == -1:
                    labels[j] = cluster
                k += 1
            cluster += 1

        self.labels_ = labels
        return self

    def _region(self, X, i):
        d = np.linalg.norm(X - X[i], axis=1)
        return np.where(d <= self.eps)[0].tolist()

    def fit_predict(self, X):
        return self.fit(X).labels_
