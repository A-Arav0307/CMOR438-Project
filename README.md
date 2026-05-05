# CMOR 438 Project

Coursework for CMOR 438 (Data Science and Machine Learning), Rice, Spring 2026.

Everything runs on the California housing dataset. The notebooks under `notebooks/` are the assignments — each one walks through a model end to end (sklearn baseline, hyperparameter sweeps, from-scratch implementation, plots). The reusable bits live in `src/rice_ml/` so the notebooks aren't 800 lines of boilerplate.

## Layout

```
src/rice_ml/                    from-scratch implementations
  supervised_ml/                  linear regression, logistic regression,
                                  perceptron, MLP, KNN, decision tree,
                                  regression tree, random forest,
                                  bagging / AdaBoost / GBM / voting,
                                  distance metrics
  unsupervised_ml/                PCA, K-Means, DBSCAN
  preprocessing/                  standardize, minmax_scale, train_test_split
  measures/                       accuracy, MSE, MAE, R², confusion matrix
tests/                          pytest suite for the src package
notebooks/
  Supervised Machine Learning/    regression + classification
  Unsupervised Machine Learning/  PCA, K-Means, DBSCAN
```

The `Data_Science_and_Machine_Learning_Spring_2022/` and `Machine-Learning-and-Data-Analytics/` folders are reference material from prior offerings and aren't part of my work.

## Running things

```
pip install -e ".[test]"
pytest
jupyter notebook
```

Python 3.9+. The notebooks expect `housing.csv` sitting next to them, which is already in each folder.

## Using the package

Every model follows the same `fit` / `predict` / `score` interface:

```python
from rice_ml.supervised_ml import LogisticRegression
from rice_ml.preprocessing import standardize, train_test_split

X = standardize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=42)

clf = LogisticRegression(lr=0.1, epochs=500).fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

Every notebook in `notebooks/` does the same thing with a different model and on the housing data.

## Notes

- `pyproject.toml` pins the package layout — `pythonpath = ["src"]` is what lets `from rice_ml.supervised_ml import Perceptron` resolve in notebooks and tests without `pip install`.
- Tests use seeded numpy RNGs, so they're deterministic. If a run flakes, that's a real bug.
- The notebooks compare each from-scratch model against a sklearn baseline. The from-scratch versions sometimes train on a subsample because their split / region-query loops aren't vectorized — that's a speed gap, not a correctness one.
