"""Logistic regression with batch gradient descent on the cross-entropy loss."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Binary logistic regression trained by gradient descent.

    Parameters:
        lr : learning rate for the GD update
        epochs : number of full-batch passes
        seed : RNG seed for the initial weights
    """

    def __init__(self, lr=0.1, epochs=500, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, size=d)
        self.b = 0.0
        self.loss_ = []
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            self.loss_.append(loss)
            grad_w = X.T @ (p - y) / n
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
