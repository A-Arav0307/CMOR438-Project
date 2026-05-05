import pytest
from rice_ml.supervised_ml.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


def test_euclidean_3_4_5():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_manhattan():
    assert manhattan_distance([1, 2], [4, 6]) == 7.0


def test_chebyshev():
    assert chebyshev_distance([0, 0], [3, 4]) == 4.0


def test_minkowski_p2_matches_euclidean():
    assert minkowski_distance([0, 0], [3, 4], p=2) == pytest.approx(5.0)


def test_minkowski_p1_matches_manhattan():
    assert minkowski_distance([1, 2], [4, 6], p=1) == pytest.approx(7.0)


def test_zero_distance_when_equal():
    p = [1.5, -2.0, 3.7]
    assert euclidean_distance(p, p) == 0.0
    assert manhattan_distance(p, p) == 0.0
    assert chebyshev_distance(p, p) == 0.0


def test_invalid_minkowski_p_raises():
    with pytest.raises(ValueError):
        minkowski_distance([0, 0], [1, 1], p=0.5)
