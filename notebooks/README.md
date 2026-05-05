# Notebooks

End-to-end walkthroughs of every algorithm on the California housing dataset.

Each notebook follows the same shape: data inspection, preprocessing, sklearn baseline, hyperparameter sweeps, a from-scratch run using `rice_ml`, and plots / metrics. The from-scratch sections import directly from the `rice_ml` package — see `src/rice_ml/` for the implementations.

```
Supervised Machine Learning/
  1. Regression/
    1. Linear Regression       normal equation + GD, Ridge / LASSO / Elastic Net
    2. Gradient Descent        learning-rate sweep, batch / SGD / mini-batch
    3. SVM Regression          linear / poly / RBF / sigmoid kernels
    4. Regression Tree         depth + criterion sweeps, feature importance
  2. Classification/
    1. Perceptron              from-scratch, decision boundary
    2. Logistic Regression     C sweep, ROC + threshold sweep
    3. Multilayer Perceptron   architecture + activation sweeps
    4. Decision Tree           depth + criterion sweeps, tree visualization
    5. Random Forest           n_estimators + depth sweeps, OOB
    6. Ensemble Methods        Bagging, AdaBoost, GBM, Voting
    7. K-Nearest Neighbors     k sweep, distance metrics

Unsupervised Machine Learning/
  1. PCA                       variance ratio, 2D/3D projection, reconstruction error
  2. K-Means                   elbow + silhouette, geographic view
  3. DBSCAN                    k-distance plot, eps + min_samples sweeps
```

All targets reframed onto the same data: regression notebooks predict `median_house_value`, classification notebooks predict whether a block is above the overall median price, unsupervised notebooks drop the target and cluster on the features only.
