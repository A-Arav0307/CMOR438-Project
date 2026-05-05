import numpy as np

from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",
                 min_samples_split=2, criterion="gini", seed=42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        if self.max_features == "sqrt":
            mf = max(1, int(np.sqrt(d)))
        elif self.max_features == "log2":
            mf = max(1, int(np.log2(d))) if d > 1 else 1
        elif isinstance(self.max_features, int):
            mf = max(1, min(self.max_features, d))
        else:
            mf = d

        self.classes_ = np.unique(y)
        self.trees_ = []
        self.feature_indices_ = []

        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            feats = rng.choice(d, size=mf, replace=False)
            t = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                seed=int(rng.integers(0, 1_000_000)),
            )
            t.fit(X[idx][:, feats], y[idx])
            self.trees_.append(t)
            self.feature_indices_.append(feats)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        c2i = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        for tree, feats in zip(self.trees_, self.feature_indices_):
            preds = tree.predict(X[:, feats])
            for j, p in enumerate(preds):
                votes[j, c2i[p]] += 1
        return self.classes_[votes.argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
