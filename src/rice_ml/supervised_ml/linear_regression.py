import numpy as np


class LinearRegression:
    def __init__(self, method="normal", lr=0.01, epochs=1000, seed=42):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xb = np.c_[np.ones(n), X]

        if self.method == "normal":
            self.theta_ = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
            self.loss_ = []
        else:
            rng = np.random.default_rng(self.seed)
            self.theta_ = rng.normal(0, 0.01, size=d + 1)
            self.loss_ = []

            for _ in range(self.epochs):
                pred = Xb @ self.theta_
                err = pred - y
                self.loss_.append(float(np.mean(err ** 2)))
                grad = (2 / n) * (Xb.T @ err)
                self.theta_ -= self.lr * grad

        self.intercept_ = float(self.theta_[0])
        self.coef_ = self.theta_[1:].copy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
