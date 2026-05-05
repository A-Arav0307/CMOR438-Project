"""Ensemble methods: Bagging, AdaBoost, Gradient Boosting, and Voting."""
import numpy as np

from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier
from rice_ml.supervised_ml.regression_tree import RegressionTree


class BaggingClassifier:
    """Bagged decision trees (bootstrap + majority vote).

    Parameters:
        n_estimators : number of bagged trees
        max_depth : maximum depth per tree
        seed : RNG seed for the bootstraps
    """

    def __init__(self, n_estimators=50, max_depth=None, seed=42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            t = DecisionTreeClassifier(
                max_depth=self.max_depth,
                seed=int(rng.integers(0, 1_000_000)),
            )
            t.fit(X[idx], y[idx])
            self.estimators_.append(t)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        c2i = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        for t in self.estimators_:
            for j, p in enumerate(t.predict(X)):
                votes[j, c2i[p]] += 1
        return self.classes_[votes.argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class AdaBoostClassifier:
    """AdaBoost with depth-1 decision stumps as the base learner (binary only).

    Parameters:
        n_estimators : number of boosting rounds
        seed : RNG seed for the weighted bootstrap each round
    """

    def __init__(self, n_estimators=50, seed=42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost only supports binary problems here")

        y_signed = np.where(y == self.classes_[1], 1, -1)
        w = np.ones(n) / n
        rng = np.random.default_rng(self.seed)

        self.estimators_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True, p=w)
            stump = DecisionTreeClassifier(max_depth=1, seed=int(rng.integers(0, 1_000_000)))
            stump.fit(X[idx], y[idx])

            preds = stump.predict(X)
            preds_signed = np.where(preds == self.classes_[1], 1, -1)

            err = float(np.sum(w * (preds_signed != y_signed)))
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)

            w = w * np.exp(-alpha * y_signed * preds_signed)
            w = w / w.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        agg = np.zeros(len(X))
        for stump, alpha in zip(self.estimators_, self.alphas_):
            preds = stump.predict(X)
            preds_signed = np.where(preds == self.classes_[1], 1, -1)
            agg += alpha * preds_signed
        return np.where(agg >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class GradientBoostingClassifier:
    """Gradient boosting for binary classification using regression trees on log-odds residuals.

    Parameters:
        n_estimators : number of boosting rounds
        lr : shrinkage applied to each tree's contribution
        max_depth : depth of each regression tree
        seed : RNG seed for tie-breaking in the trees
    """

    def __init__(self, n_estimators=100, lr=0.1, max_depth=3, seed=42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier only supports binary problems")

        y01 = np.where(y == self.classes_[1], 1.0, 0.0)
        p0 = float(np.clip(y01.mean(), 1e-6, 1 - 1e-6))
        self.f0_ = float(np.log(p0 / (1 - p0)))

        F = np.full(len(X), self.f0_)
        rng = np.random.default_rng(self.seed)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))
            residuals = y01 - p
            tree = RegressionTree(max_depth=self.max_depth, seed=int(rng.integers(0, 1_000_000)))
            tree.fit(X, residuals)
            F = F + self.lr * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.f0_)
        for t in self.estimators_:
            F = F + self.lr * t.predict(X)
        p1 = 1.0 / (1.0 + np.exp(-F))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class VotingClassifier:
    """Combine several fitted classifiers by hard or soft voting.

    Parameters:
        estimators : list of (name, estimator) tuples
        voting : 'hard' for majority vote on predicted labels, 'soft' to average predicted probabilities
    """

    def __init__(self, estimators, voting="hard"):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        for _, est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.voting == "hard":
            c2i = {c: i for i, c in enumerate(self.classes_)}
            votes = np.zeros((len(X), len(self.classes_)), dtype=int)
            for _, est in self.estimators:
                for j, p in enumerate(est.predict(X)):
                    votes[j, c2i[p]] += 1
            return self.classes_[votes.argmax(axis=1)]

        agg = np.zeros((len(X), len(self.classes_)))
        for _, est in self.estimators:
            if not hasattr(est, "predict_proba"):
                raise ValueError("soft voting requires predict_proba on every estimator")
            agg += est.predict_proba(X)
        return self.classes_[agg.argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
