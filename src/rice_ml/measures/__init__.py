import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    L = len(labels)

    idx_true = np.searchsorted(labels, y_true)
    idx_pred = np.searchsorted(labels, y_pred)
    M = np.zeros((L, L), dtype=int)
    for t, p in zip(idx_true, idx_pred):
        M[t, p] += 1
    return M


__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "confusion_matrix",
]
