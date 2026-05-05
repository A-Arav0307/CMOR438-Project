# DBSCAN

Density-based clustering. DBSCAN doesn't take $k$ — it discovers the number of clusters from the data, and importantly it labels low-density points as **noise** instead of forcing them into a cluster.

## How it works

Two hyperparameters: `eps` (neighborhood radius) and `min_samples` (how many points are needed to form a dense region).

- A point is a **core point** if at least `min_samples` other points are within `eps` of it
- Two core points within `eps` of each other are in the same cluster
- A non-core point near a core point is a **border point** and joins that cluster
- Points near no core point are **noise**

The algorithm grows clusters by region-query expansion from each unvisited core point. There's no centroid step — the clusters can be arbitrarily shaped.

## Picking eps

The classic heuristic: plot each point's distance to its $k$-th nearest neighbor (sorted ascending). Look for the "knee" of that curve — that's a sensible `eps`. Below it, you over-fragment; above it, you over-merge.

## Sections

1. Same preprocessing pipeline
2. Drop the target
3. **k-distance plot** to pick `eps`
4. Default-ish baseline at the knee value
5. **`eps` sweep** — number of clusters and noise-point fraction at each value
6. **`min_samples` sweep** — same idea
7. **PCA-projected scatter**, colored by cluster (noise = grey)
8. **Geographic scatter** — the most informative one, since DBSCAN naturally finds population-density clusters (essentially: cities)
9. **Geography-only** run — fit DBSCAN on `[latitude, longitude]` alone. The cluster assignments correspond closely to the actual cities of California.
10. From-scratch `rice_ml.DBSCAN` on a 2,000-row subsample (no kd-tree, brute-force region queries)

## Notes

DBSCAN's behavior on this dataset is qualitatively different from K-Means: instead of partitioning all points, it finds dense regions and treats the rest as noise. About 10–25% of points get labeled as noise depending on `eps`, which is a useful signal in itself — those are the truly rural blocks far from any population center.
