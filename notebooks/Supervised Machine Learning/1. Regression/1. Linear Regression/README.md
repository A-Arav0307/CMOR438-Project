# Linear Regression

Predicting `median_house_value` on the California housing data.

The notebook fits the model two ways — gradient descent and the normal equation — and compares them. Preprocessing handles the missing `total_bedrooms` (median fill), one-hot encodes `ocean_proximity`, and z-scores the numeric columns.

Sections: data inspection, feature/target plots, preprocessing, train/test split, GD vs normal equation, predicted-vs-actual, metrics, Ridge/LASSO/Elastic Net, and a sanity check against my `rice_ml.LinearRegression` class.

`housing.csv` lives next to the notebook.
