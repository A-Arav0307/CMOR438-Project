import numpy as np
import pytest
from rice_ml.supervised_ml.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier
from rice_ml.supervised_ml.perceptron import Perceptron


def test_bagging(linearly_separable):
    X, y = linearly_separable
    bag = BaggingClassifier(n_estimators=10, max_depth=2).fit(X, y)
    assert bag.score(X, y) > 0.9


def test_adaboost(linearly_separable):
    X, y = linearly_separable
    ada = AdaBoostClassifier(n_estimators=20, seed=0).fit(X, y)
    assert ada.score(X, y) > 0.9


def test_gradient_boosting(linearly_separable):
    X, y = linearly_separable
    gb = GradientBoostingClassifier(n_estimators=20, lr=0.1, max_depth=2, seed=0).fit(X, y)
    assert gb.score(X, y) > 0.9


def test_gradient_boosting_proba_shape(linearly_separable):
    X, y = linearly_separable
    gb = GradientBoostingClassifier(n_estimators=10, max_depth=2).fit(X, y)
    proba = gb.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_voting_hard(linearly_separable):
    X, y = linearly_separable
    estimators = [
        ("p", Perceptron(epochs=10)),
        ("t", DecisionTreeClassifier(max_depth=3)),
    ]
    v = VotingClassifier(estimators=estimators, voting="hard").fit(X, y)
    assert v.score(X, y) > 0.9


def test_invalid_voting_raises():
    with pytest.raises(ValueError):
        VotingClassifier(estimators=[], voting="bogus")
