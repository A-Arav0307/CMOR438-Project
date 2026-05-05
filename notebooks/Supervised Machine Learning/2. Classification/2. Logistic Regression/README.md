# Logistic Regression

Same above-median binary target as the perceptron notebook, but fitted as a probabilistic model.

## The model

Logistic regression predicts $P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ where $\sigma$ is the sigmoid. It's fit by minimizing the cross-entropy loss:

$$L = -\frac{1}{n} \sum_i \big[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \big]$$

Convex in $\mathbf{w}$, so gradient descent converges to the global minimum (no learning rate / initialization headache like the perceptron).

## Regularization

sklearn's `C` is the inverse of the L2 penalty strength: small `C` ⇒ strong regularization, large `C` ⇒ closer to unregularized. The sweep walks `C ∈ {0.01, 0.1, 1, 10, 100}` and plots train + test accuracy. The gap between them is the overfitting story.

## Sections

1. Reframe target, same preprocessing
2. sklearn baseline at default `C=1.0`
3. Regularization sweep
4. From-scratch GD implementation from `rice_ml.LogisticRegression`
5. ROC curve + AUC
6. Threshold sweep — vary the decision threshold from 0 to 1, plot precision and recall against it. Default 0.5 is rarely actually optimal.
7. Top coefficients — sort the learned weights by absolute value to see which features push the prediction up or down. `median_income` dominates positively; `INLAND` dominates negatively, both expected.

## Notes

Test accuracy lands around **82–84%** — a clear improvement over the perceptron's ~76%. Most of the gain is from the smooth loss letting the optimizer find a better boundary, not from any added expressive power.
