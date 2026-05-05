# K-Means

Cluster the housing features into $k$ groups by minimizing within-cluster squared distance:

$$J = \sum_{i=1}^{n} \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\|^2$$

Lloyd's algorithm alternates: assign each point to the nearest centroid, then recompute centroids as the cluster mean. Repeat until assignments stop changing.

## Choosing $k$

Two complementary methods:

- **Elbow method** — plot inertia $J$ against $k$. It strictly decreases, but the rate of decrease changes at the "right" $k$.
- **Silhouette score** — measures how well each point fits its cluster vs the next-best cluster. Range [-1, 1], higher is better.

Both are heuristics — they often disagree by one or two on $k$. The notebook plots both and picks a value that's reasonable on both.

## Sections

1. Same preprocessing pipeline
2. Drop the target
3. $k=5$ baseline
4. **Elbow + silhouette sweep** over `k ∈ {2, ..., 10}`
5. **PCA-projected scatter** of the chosen-$k$ clusters
6. **Geographic scatter** by lat/long, colored by cluster — the most useful visualization, since California has clear geographic structure (Bay Area, LA, Central Valley, far north)
7. **Target-vs-cluster** check — for each cluster, mean and median of the held-out price. If the clusters line up with price quartiles, K-Means found something real.
8. From-scratch `rice_ml.KMeans` against sklearn — same number of clusters, same data, expect very similar centroids modulo permutation

## Notes

K-Means is sensitive to initialization — the algorithm converges to a local minimum, and bad init can land far from the global one. sklearn defaults to `k-means++` (smart init); my from-scratch version uses uniform random init and runs multiple restarts to compensate.
