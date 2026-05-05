"""One-hidden-layer MLP with ReLU hidden + sigmoid output, trained by mini-batch SGD."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


class MLP:
    """Single-hidden-layer neural net for binary classification.

    Parameters:
        hidden : number of units in the hidden layer
        lr : learning rate for SGD
        epochs : number of passes over the training data
        batch : mini-batch size
        seed : RNG seed for weight initialization
    """

    def __init__(self, hidden=32, lr=0.05, epochs=50, batch=64, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        h = self.hidden
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), size=(d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h), size=(h, 1))
        self.b2 = np.zeros(1)
        self.loss_ = []
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch):
                b_idx = idx[s:s + self.batch]
                xb, yb = X[b_idx], y[b_idx]
                z1 = xb @ self.W1 + self.b1
                a1 = relu(z1)
                z2 = a1 @ self.W2 + self.b2
                p = sigmoid(z2)
                loss = -np.mean(yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))
                ep_loss += loss * len(b_idx)
                dz2 = (p - yb) / len(b_idx)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * relu_grad(z1)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            self.loss_.append(ep_loss / n)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        a1 = relu(X @ self.W1 + self.b1)
        return sigmoid(a1 @ self.W2 + self.b2).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
