# Unsupervised Machine Learning

California housing dataset with the target dropped — these notebooks cluster and project on features alone, then check after the fact whether structure in the features lines up with the price target.

## Notebook pattern

1. **Data inspection** — same as the supervised notebooks (shapes, dtypes, nulls)
2. **Preprocessing** — median fill, one-hot encode `ocean_proximity`, z-score numeric features. Standardization matters more here than in the supervised notebooks because every method below is distance- or variance-based.
3. **Drop the target** — `median_house_value` is set aside, not fed in
4. **sklearn baseline** — fit the reference implementation
5. **Hyperparameter sweep** — `n_components` for PCA, `k` for K-Means, `eps` / `min_samples` for DBSCAN
6. **From-scratch (rice_ml)** — same algorithm, my implementation
7. **Visualization** — PCA scatter, geographic (lat/long) scatter colored by cluster, target overlay
8. **Sanity check vs target** — does the structure we found correspond to anything meaningful in the held-out price?

## Notebooks

- **1. PCA** — explained variance ratio (scree plot), cumulative variance, 2D and 3D projections colored by price, component loadings (which original features each PC weights), reconstruction error vs `n_components`. From-scratch via covariance eigendecomposition.
- **2. K-Means** — elbow method on inertia, silhouette score sweep, both PCA-projected and geographic (lat/long) visualizations of the clusters, mean target per cluster as a sanity check that the clusters are picking up something price-relevant. From-scratch Lloyd's algorithm.
- **3. DBSCAN** — k-distance plot to pick `eps`, sweeps over `eps` and `min_samples`, noise-point counts, PCA + geographic views. No from-scratch comparison runs on the full dataset since the region-query loop isn't vectorized — the from-scratch version runs on a subsample and gets the same cluster structure.

## Why drop the target

The point of unsupervised learning is finding structure without labels. Comparing cluster assignments to the held-out price afterward is a fairness check on whether the structure is meaningful, not a training signal. K-Means clusters that happen to align with price quartiles are a more interesting result than ones that don't, but neither outcome changes how the model was fit.
