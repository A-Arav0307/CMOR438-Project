import numpy as np
import pytest
from rice_ml.measures import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)


def test_accuracy_perfect():
    y = np.array([1, 0, 1, 1, 0])
    assert accuracy_score(y, y) == 1.0


def test_accuracy_zero():
    y = np.array([0, 0, 0, 0])
    p = np.array([1, 1, 1, 1])
    assert accuracy_score(y, p) == 0.0


def test_mse():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 4.0])
    assert mean_squared_error(y, p) == pytest.approx(1.0 / 3)


def test_mae():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 4.0, 6.0])
    assert mean_absolute_error(y, p) == 2.0


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0


def test_r2_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.full(4, y.mean())
    assert r2_score(y, p) == 0.0


def test_confusion_matrix():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    M = confusion_matrix(y_true, y_pred)
    expected = np.array([[1, 1], [1, 2]])
    assert np.array_equal(M, expected)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy_score([1, 0], [1, 0, 1])
