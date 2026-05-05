# DBSCAN

Density-based clustering on the same housing feature set.

Baseline, then a k-distance plot to pick a sensible `eps`, followed by sweeps over `eps` and `min_samples`. Same two visualizations as the K-Means notebook (PCA projection + geographic view), plus a separate run on geography only — DBSCAN does interesting things when you let it cluster cities directly on lat/long.
