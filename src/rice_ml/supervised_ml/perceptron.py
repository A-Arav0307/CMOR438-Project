import numpy as np


class Perceptron:
    def __init__(self, eta=0.01, epochs=50, seed=42):
        self.eta = eta
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, size=d)
        self.b = 0.0
        y_signed = np.where(y == 1, 1, -1)
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for i in range(n):
                pred = 1 if (X[i] @ self.w + self.b) >= 0 else -1
                update = self.eta * (y_signed[i] - pred)
                if update != 0:
                    self.w += update * X[i]
                    self.b += update
                    errors += 1
            self.errors_.append(errors)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.w + self.b >= 0).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
