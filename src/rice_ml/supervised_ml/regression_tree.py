import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def _var(y):
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2, seed=42):
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree_ = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or np.var(y) < 1e-12):
            return _Node(value=float(y.mean()))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return _Node(value=float(y.mean()))

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            return _Node(value=float(y.mean()))

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feat, threshold=thr, left=left, right=right)

    def _best_split(self, X, y):
        n, d = X.shape
        parent = _var(y)
        best_gain = 0.0
        best_feat, best_thr = None, None
        for feat in range(d):
            values = np.unique(X[:, feat])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                w_l = len(left) / n
                w_r = len(right) / n
                gain = parent - (w_l * _var(left) + w_r * _var(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr
        return best_feat, best_thr

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._descend(x, self.tree_) for x in X])

    def _descend(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._descend(x, node.left)
        return self._descend(x, node.right)

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
