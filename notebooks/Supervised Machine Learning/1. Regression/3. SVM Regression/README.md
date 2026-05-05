# SVM Regression

SVR with linear, poly, RBF, and sigmoid kernels on the housing data.

Heads up: the training set is subsampled because full SVR on 13k rows is painfully slow. Notebook compares predicted-vs-actual per kernel, picks the best kernel feature-by-feature, plots error distributions, and finishes with a 3D geographic view of residuals.
