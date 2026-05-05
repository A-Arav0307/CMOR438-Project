# PCA

Principal component analysis on the housing features (the price target is dropped — this is unsupervised). PCA finds an orthogonal basis where the first axis captures the most variance, the second captures the most variance subject to being orthogonal to the first, and so on.

## How it works

Standardize the features, compute the covariance matrix $\Sigma = \frac{1}{n} X^\top X$, take its eigendecomposition. The eigenvectors are the principal components; the eigenvalues are how much variance each one explains:

$$\text{explained variance ratio}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$

Project the data onto the top $k$ components by multiplying $X$ by the matrix of those $k$ eigenvectors.

## Sections

1. Same preprocessing as the supervised notebooks (median fill, one-hot, z-score)
2. Drop the target
3. Full PCA fit, all components
4. **Scree plot** — explained variance per component, cumulative variance overlaid. Look for the elbow (~3–4 components capture most of it on this dataset)
5. **2D projection** — scatter of PC1 vs PC2, colored by the held-out target as a sanity check
6. **3D projection** — PC1 / PC2 / PC3 scatter
7. **Component loadings** — each PC is a linear combination of the original features; print which originals dominate each PC. PC1 is mostly "size of the block" (rooms / population / households), PC2 is mostly geographic (lat / long).
8. **Reconstruction error** vs `n_components` — project and re-project, measure squared error. Confirms the elbow visually.
9. From-scratch `rice_ml.PCA` against sklearn

## Notes

PCA is the cheapest "is there structure here" test you can run. The fact that PC1 cleanly separates the price target despite never seeing the target is a strong sign the housing features carry real signal — that's why all the supervised models in the sibling folder do reasonably well.
