import numpy as np
import pytest
from rice_ml.unsupervised_ml import PCA


def test_invalid_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)


def test_captures_dominant_direction(correlated_2d):
    X = correlated_2d
    pca = PCA(n_components=1).fit(X)
    assert pca.explained_variance_ratio_[0] > 0.98


def test_transform_shape(correlated_2d):
    pca = PCA(n_components=1).fit(correlated_2d)
    Z = pca.transform(correlated_2d)
    assert Z.shape == (correlated_2d.shape[0], 1)


def test_inverse_transform_approximate(correlated_2d):
    pca = PCA(n_components=2).fit(correlated_2d)
    Z = pca.transform(correlated_2d)
    X_rec = pca.inverse_transform(Z)
    assert np.allclose(X_rec, correlated_2d, atol=1e-8)


def test_variance_ratio_sums_to_one(correlated_2d):
    pca = PCA(n_components=2).fit(correlated_2d)
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-9
