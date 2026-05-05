# Multilayer Perceptron

Same above-median target. Now we stack hidden layers — each layer is a linear transformation followed by a nonlinearity, and the network can express boundaries that no single linear model can.

## The model

Forward pass for a one-hidden-layer network:

$$\mathbf{h} = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{p} = \sigma(W_2 \mathbf{h} + b_2)$$

Trained by minimizing cross-entropy via backpropagation — gradients flow back through the chain rule, layer weights get updated by mini-batch SGD.

## Architecture sweep

The sklearn baseline uses `hidden_layer_sizes=(100,)`. The notebook sweeps:

| Layout | Notes |
|---|---|
| `(50,)` | Underfit — not enough capacity |
| `(100,)` | sklearn default, the reference point |
| `(100, 50)` | Two layers, capacity comparable to (100,) |
| `(200, 100)` | Bigger, slower, slight overfit |
| `(50, 50, 50)` | Deep + narrow, more nonlinearity per parameter |

## Activation comparison

ReLU vs tanh vs logistic at fixed architecture. ReLU usually wins on tabular data because the gradient doesn't saturate.

## Sections

1. Same preprocessing pipeline as the other classification notebooks
2. sklearn `MLPClassifier` baseline
3. Architecture sweep
4. Activation comparison
5. From-scratch one-hidden-layer net with manual backprop, imported from `rice_ml.MLP`. Forward pass, loss, backward pass, mini-batch SGD all written out by hand.
6. ROC curve

## Notes

The MLP is the **best classifier on this dataset** in this project, landing around **85%** test accuracy. Decision trees and random forests get close, but the MLP edges them out because the underlying class boundary is smooth-ish and the MLP can fit smooth boundaries directly without averaging over discrete splits.
