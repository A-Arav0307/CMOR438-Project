# Supervised Machine Learning

California housing dataset.

## Regression
Target: `median_house_value` (continuous).

- **1. Linear Regression** — normal equation + GD, Ridge, LASSO, Elastic Net
- **2. Gradient Descent** — learning rate sweep, batch/SGD/mini-batch
- **3. SVM Regression** — linear, poly, RBF, sigmoid kernels

## Classification
Target: `median_house_value > median` (binary).

- **1. Perceptron** — sklearn baseline, learning rate sweep, from-scratch, decision boundary
- **2. Logistic Regression** — baseline, C sweep, from-scratch GD, ROC, coefficients
- **3. Multilayer Perceptron** — baseline, architecture/activation sweeps, from-scratch, ROC
