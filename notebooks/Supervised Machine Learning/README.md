# Supervised Machine Learning

California housing dataset, used twice over: continuous target for the regression notebooks, thresholded binary target for the classification notebooks.

## Notebook pattern

Every notebook in here follows the same shape:

1. **Data inspection** — shapes, dtypes, nulls, basic describe
2. **Feature vs target plots** — scatter / box plots to spot signal
3. **Preprocessing** — median fill on `total_bedrooms`, one-hot encode `ocean_proximity`, z-score the numeric features
4. **Train/test split** — 67/33, fixed `random_state=42`
5. **sklearn baseline** — default settings as the reference
6. **Hyperparameter sweep(s)** — depth, learning rate, k, etc.
7. **From-scratch (rice_ml)** — same model, same data, my own implementation imported from the package
8. **Diagnostics** — predicted vs actual, ROC curve, feature importance, etc.

## Regression
Target: `median_house_value` (continuous).

- **1. Linear Regression** — normal equation + GD, Ridge, LASSO, Elastic Net. Tests R² on the test set; expect ~0.64.
- **2. Gradient Descent** — learning rate sweep, batch / SGD / mini-batch comparison on the same problem.
- **3. SVM Regression** — linear, polynomial, RBF, sigmoid kernels. Subsamples first since SVR is `O(n²)`–`O(n³)` to train.
- **4. Regression Tree** — depth sweep, splitting criterion comparison, feature importance, shallow tree visualization.

## Classification
Target: `median_house_value > median` (binary).

- **1. Perceptron** — sklearn baseline, learning rate sweep, from-scratch, decision boundary on two features.
- **2. Logistic Regression** — baseline, regularization (C) sweep, from-scratch GD, ROC + threshold sweep, top coefficients.
- **3. Multilayer Perceptron** — baseline, architecture sweep, activation comparison, from-scratch backprop, ROC. Best classifier on this dataset (~85%).
- **4. Decision Tree** — depth sweep showing overfitting in real time, criterion comparison (gini / entropy / log_loss), feature importance, depth-3 visualization.
- **5. Random Forest** — `n_estimators` and depth sweeps, feature importance, OOB score, ROC.
- **6. Ensemble Methods** — Bagging, AdaBoost, Gradient Boosting, Voting (hard + soft). Includes a learning-rate sweep on GBM.
- **7. K-Nearest Neighbors** — k sweep, distance metric (Euclidean / Manhattan / Chebyshev) and weighting comparison, from-scratch, decision boundary on two features, ROC.

## Why one dataset throughout

It would be easier to get good results by picking a different dataset for each algorithm (Pima diabetes for logistic regression, iris for KNN, blobs for clustering). Using one real dataset everywhere is harder but more honest — the comparisons across algorithms are direct because the data is the same.
