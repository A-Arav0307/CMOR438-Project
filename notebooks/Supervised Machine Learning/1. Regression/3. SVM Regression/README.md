# SVM Regression

Support vector regression on `median_house_value`. SVR fits a tube of width $\epsilon$ around the regression function and only penalizes points outside it:

$$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, |y_i - \mathbf{w}^\top \phi(\mathbf{x}_i)| - \epsilon)$$

The $\phi$ is wherever the kernel trick comes in — different kernels mean different feature spaces.

## Heads up: subsampled

Full SVR on 13k training rows takes minutes-to-hours because training is $O(n^2)$ to $O(n^3)$. This notebook subsamples to 2,000 rows for the kernel sweep. The sklearn baseline runs on the same subsample so the numbers are comparable.

## Kernels compared

| Kernel | $K(\mathbf{x}, \mathbf{x}')$ | Notes |
|---|---|---|
| Linear | $\mathbf{x}^\top \mathbf{x}'$ | Should match linear regression closely |
| Polynomial | $(\gamma \mathbf{x}^\top \mathbf{x}' + r)^d$ | Expands feature crosses up to degree d |
| RBF | $\exp(-\gamma \|\mathbf{x}-\mathbf{x}'\|^2)$ | Local — usually the best on tabular data |
| Sigmoid | $\tanh(\gamma \mathbf{x}^\top \mathbf{x}' + r)$ | Mostly for completeness, rarely the best |

## Sections

1. Same preprocessing pipeline
2. Subsample + train one SVR per kernel
3. Predicted-vs-actual scatter per kernel, side by side
4. Per-feature breakdown — which kernel does best when you regress price against just one feature
5. Error distribution per kernel
6. 3D geographic scatter of residuals (lat, long, residual) to see where the model fails geographically

The geographic residual plot is the most informative — RBF still has structured residuals along the coast, which says even a flexible kernel hasn't fully captured the "near the ocean" effect that one-hot `ocean_proximity` is trying to pick up.
