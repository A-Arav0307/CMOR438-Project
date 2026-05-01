# CMOR 438 Project

Coursework for CMOR 438 (Data Science and Machine Learning), Rice, Spring 2026.

Everything runs on the California housing dataset. The notebooks under `Supervised Machine Learning/` and `Unsupervised Machine Learning/` are the assignments — each one walks through a model end to end (sklearn baseline, hyperparameter sweeps, from-scratch implementation, plots). The reusable bits live in `src/rice_ml/` so the notebooks aren't 800 lines of boilerplate.

## Layout

```
src/rice_ml/                    from-scratch implementations
  supervised_ml/                  perceptron, logistic regression, MLP, KNN
  unsupervised_ml/                PCA, KMeans
  preprocessing/, measures/
tests/                          pytest suite for the src package
Supervised Machine Learning/    notebooks (regression + classification)
Unsupervised Machine Learning/  notebooks (PCA, K-Means, DBSCAN)
```

The `Data_Science_and_Machine_Learning_Spring_2022/` and `Machine-Learning-and-Data-Analytics/` folders are reference material from prior offerings and aren't part of my work.

## Running things

```
pip install -e ".[test]"
pytest
jupyter notebook
```

Python 3.9+. The notebooks expect the `housing.csv` sitting next to them, which is already there.

## Notes

- `pyproject.toml` pins the package layout — `pythonpath = ["src"]` is what lets `from rice_ml.supervised_ml import Perceptron` resolve in the notebooks and tests.
- Tests use seeded numpy RNGs, so they're deterministic. If a run flakes, that's a real bug.
