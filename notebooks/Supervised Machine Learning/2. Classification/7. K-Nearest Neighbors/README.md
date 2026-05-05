# K-Nearest Neighbors

KNN doesn't learn parameters — it memorizes the training set and at prediction time looks up the $k$ closest training points and votes. "Closest" depends on the distance metric.

## Distance metrics compared

| Metric | Formula |
|---|---|
| Euclidean | $\sqrt{\sum_i (x_i - x'_i)^2}$ |
| Manhattan | $\sum_i \|x_i - x'_i\|$ |
| Chebyshev | $\max_i \|x_i - x'_i\|$ |
| Minkowski-p | $(\sum_i \|x_i - x'_i\|^p)^{1/p}$ — generalizes the above |

Standardization matters a lot here — without z-scoring, `total_rooms` (range ~0–40k) would swamp `median_income` (range ~0–15) in the distance.

## Weighting

`uniform` — every neighbor counts equally. `distance` — closer neighbors get more vote. With `distance` weighting the choice of $k$ matters less, since far neighbors barely contribute.

## Sections

1. Same above-median preprocessing
2. sklearn `KNeighborsClassifier` baseline at $k=5$
3. **k sweep** — `k ∈ {1, 3, 5, 10, 20, 50, 100}`. $k=1$ overfits hard, $k=100$ underfits. There's a sweet spot.
4. **Metric comparison** at the chosen $k$ — euclidean / manhattan / chebyshev / minkowski-p
5. **Weighting comparison** — uniform vs distance
6. From-scratch brute-force KNN from `rice_ml.KNN` (no kd-tree)
7. Decision boundary on two features (`median_income`, `latitude`) — KNN's boundary is locally adaptive in a way no parametric model in this project is
8. ROC curve

## Notes

KNN sits a step behind the MLP and forest at around **80–82%** accuracy. The decision boundary plot is the most striking output — it has the wiggly, locally-adaptive look that's the visual signature of KNN.
