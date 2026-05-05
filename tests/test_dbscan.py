import numpy as np
from rice_ml.unsupervised_ml.dbscan import DBSCAN


def test_recovers_blobs(blobs):
    db = DBSCAN(eps=1.5, min_samples=5).fit(blobs)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    assert n_clusters == 3


def test_noise_point_labeled_minus_one():
    X = np.array([
        [0.0, 0.0], [0.05, 0.0], [0.0, 0.05],
        [50.0, 50.0],
    ])
    db = DBSCAN(eps=0.2, min_samples=2).fit(X)
    assert db.labels_[3] == -1


def test_fit_predict_shape(blobs):
    labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(blobs)
    assert labels.shape == (len(blobs),)
