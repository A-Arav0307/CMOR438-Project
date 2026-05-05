# Perceptron

First classification notebook. The target gets reframed: instead of predicting the price, predict whether a block's `median_house_value` is above the overall median. That gives a roughly 50/50 binary label and keeps the same dataset.

## The model

Rosenblatt's perceptron — the original linear classifier. Predict $\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$. On every misclassification, push the weights toward the correct answer:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i \quad \text{if } y_i (\mathbf{w}^\top \mathbf{x}_i + b) \le 0$$

If the data is linearly separable, this is guaranteed to converge. Our preprocessed housing features are *not* linearly separable, so the perceptron oscillates around its best linear boundary.

## Sections

1. Reframe the target: `y = (price > price.median()).astype(int)`
2. Same preprocessing pipeline as the regression notebooks
3. sklearn `Perceptron` baseline
4. Learning rate sweep, look at how the accuracy curve flattens out
5. From-scratch implementation imported from `rice_ml.supervised_ml.Perceptron`
6. Decision boundary plot on two features (`median_income`, `latitude`) — fits a perceptron on just those two columns so we can actually visualize what it learned

## Notes

The perceptron lands around **75–78%** accuracy here. That's a useful baseline but logistic regression (next notebook) will beat it because it has a smooth loss and can express uncertainty via probabilities. The decision boundary plot makes the linearity limitation visceral — the actual class boundary is curved, the perceptron is forced to draw a straight line.
