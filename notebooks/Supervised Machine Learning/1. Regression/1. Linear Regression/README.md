# Linear Regression

Predicting `median_house_value` on the California housing data. This is the foundational notebook — it sets up the preprocessing pipeline that every other regression notebook reuses.

## The model

Linear regression assumes the target is a weighted sum of the features:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

The weights $\mathbf{w}$ and bias $b$ are chosen to minimize the sum of squared residuals. There are two ways to do that here: the closed-form **normal equation** $\mathbf{w} = (X^\top X)^{-1} X^\top y$ (one shot, exact), and **gradient descent** (iterative, scales better but needs a learning rate). The notebook fits both and confirms they land on the same coefficients up to numerical noise.

## Sections

1. Data inspection — 20,640 rows, 9 features, one categorical (`ocean_proximity`), 207 nulls in `total_bedrooms`
2. Feature/target plots — scatter `median_income` against price, the strongest signal
3. Preprocessing — median fill, one-hot encode, z-score numerics
4. Train/test split — 67/33, `random_state=42`
5. Normal equation fit + R²
6. Gradient descent fit + loss curve
7. Predicted-vs-actual scatter, residual histogram
8. Regularization: Ridge, LASSO, Elastic Net at a small grid of alpha values
9. From-scratch sanity check via `rice_ml.LinearRegression`

## What to expect

Test R² lands around **0.64**. Linear regression is a floor, not a ceiling, on this dataset — the relationships between features and price are pretty nonlinear (especially anything geographic), and the tree-based notebooks beat this by a wide margin.

`housing.csv` lives next to the notebook.
