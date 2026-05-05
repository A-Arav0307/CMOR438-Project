"""Pairwise distance functions used by KNN and friends."""
import numpy as np


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b)))


def chebyshev_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


def minkowski_distance(a, b, p=2):
    """Generalized L-p distance — p=1 is Manhattan, p=2 is Euclidean."""
    if p < 1:
        raise ValueError("p must be >= 1")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b) ** p) ** (1 / p))
