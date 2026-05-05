# Gradient Descent

Same housing regression target, but the focus shifts from the *model* to the *optimizer*. The model is still a linear regression — what changes here is how we fit it.

## The three flavors

| Variant | Update uses | Per-step cost | Behavior |
|---|---|---|---|
| **Batch GD** | All n training points | $O(nd)$ | Smooth descent, slow per epoch |
| **SGD** | One random point | $O(d)$ | Noisy path, very fast per step |
| **Mini-batch** | A batch of size B | $O(Bd)$ | Compromise — most common in practice |

All three minimize the same objective:

$$L(\mathbf{w}) = \frac{1}{2n}\sum_i (\mathbf{w}^\top \mathbf{x}_i - y_i)^2$$

with the gradient $\nabla L = \frac{1}{n} X^\top (X\mathbf{w} - \mathbf{y})$. The difference is just which rows of $X$ go into that gradient on each step.

## Sections

1. Same preprocessing pipeline as the linear regression notebook
2. Learning rate sweep — `lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}`. Too small ⇒ doesn't converge in budget. Too large ⇒ loss diverges to infinity. Look for the sweet spot.
3. Batch / SGD / mini-batch comparison — same epochs and same lr, plot loss curves on one axis
4. Test predictions and residual histogram for the winning configuration

## Notes

The learning rate sweep is the interesting part — there's a real "edge of stability" effect where bumping lr by 10× pushes the loss from monotone decrease to NaN within a few epochs. SGD's loss curve is jaggy by construction; the smoothed version still trends down at roughly the same rate as batch GD on this problem.
