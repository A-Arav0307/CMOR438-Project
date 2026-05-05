# K-Means

Clustering the housing features without using the price.

k=5 baseline first, then elbow + silhouette to argue for a value of k. Clusters get visualized two ways: projected onto the top two PCs, and laid out geographically by lat/long since this is California. After that, a target-vs-cluster check to see whether the unsupervised structure lines up with median house value, and a from-scratch K-Means matching `src/rice_ml/unsupervised_ml/kmeans.py`.
