# Ensemble Methods

Four ensembles compared on the above-median task: Bagging, AdaBoost, Gradient Boosting, and Voting (hard + soft).

Each one gets its own section with its own knobs. There's also a learning rate sweep on the GBM specifically, since it's the most sensitive to it. Final cell stacks them side-by-side.

There's also a from-scratch section at the end that runs my `rice_ml` versions of bagging, AdaBoost, and GBM against the sklearn results.
