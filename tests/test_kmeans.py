import numpy as np
import pytest
from rice_ml.unsupervised_ml import KMeans


def test_invalid_k():
    with pytest.raises(ValueError):
        KMeans(k=0)


def test_recovers_three_blobs(blobs):
    km = KMeans(k=3, max_iter=100).fit(blobs)
    assert len(km.centroids_) == 3
    counts = np.bincount(km.labels_, minlength=3)
    assert counts.min() > 50


def test_inertia_decreases(blobs):
    km = KMeans(k=3, max_iter=20).fit(blobs)
    hist = km.inertia_history_
    for a, b in zip(hist, hist[1:]):
        assert b <= a + 1e-6


def test_predict_matches_fit_labels(blobs):
    km = KMeans(k=3).fit(blobs)
    assert np.array_equal(km.predict(blobs), km.labels_)


def test_reproducibility(blobs):
    a = KMeans(k=3, seed=5).fit(blobs).labels_
    b = KMeans(k=3, seed=5).fit(blobs).labels_
    assert np.array_equal(a, b)
