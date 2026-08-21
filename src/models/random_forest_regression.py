from numpy import linspace
from sklearn.ensemble import RandomForestRegressor

from .base_regression_model import BaseRegressionModel


class RandomForestRegressionModel(BaseRegressionModel):
    def __init__(self):
        reg = RandomForestRegressor()
        max_depth = [int(x) for x in linspace(10, 110, num=11)]
        max_depth.append(None)
        param_grid = {
            "n_estimators": [int(x) for x in linspace(start=200, stop=2000, num=10)],
            "max_depth": max_depth,
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            # 1.0 (= all features) is what "auto" MEANT for a regressor; the literal was
            # removed in scikit-learn 1.3 and this project pins 1.9, so sampling it raised
            # InvalidParameterError mid-search rather than at construction.
            "max_features": [1.0, "sqrt"],
            "bootstrap": [True, False],
        }
        super().__init__(reg, param_grid)
