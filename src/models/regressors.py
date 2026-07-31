"""Classical regressor registry."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

RS = 42

# Inner CV folds for GridSearchCV hyperparameter tuning (within each outer training fold).
# 3 is safe for small battery datasets where training folds can have ~8-12 cells.
_INNER_CV = 3


def _try(name: str, factory):
    try:
        return factory()
    except Exception:
        return None


def _gs(estimator, param_grid: dict, *, scoring: str = "r2") -> GridSearchCV:
    return GridSearchCV(
        estimator,
        param_grid,
        cv=_INNER_CV,
        scoring=scoring,
        refit=True,
        n_jobs=1,
        error_score=0.0,
    )


def build_regressors() -> Dict[str, Any]:
    reg: Dict[str, Any] = {
        "DummyMean":        DummyRegressor(strategy="mean"),
        "DummyMedian":      DummyRegressor(strategy="median"),
        # The runner maps this placeholder to the causal previous-RPT value
        # (or zero change for delta targets) before statistical preprocessing.
        "PreviousRPT":      DummyRegressor(strategy="mean"),
        "Ridge":            Ridge(alpha=1.0, random_state=RS),
        "Lasso":            Lasso(alpha=0.001, random_state=RS, max_iter=10_000),
        "ElasticNet":       ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RS, max_iter=10_000),
        "SVR-linear":       SVR(kernel="linear"),
        "SVR-RBF":          SVR(kernel="rbf"),
        "KNN":              KNeighborsRegressor(n_neighbors=5),
        "DecisionTree":     DecisionTreeRegressor(random_state=RS),
        "RandomForest":     RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
        "GradientBoosting": GradientBoostingRegressor(random_state=RS),
        "MLP":              MLPRegressor(hidden_layer_sizes=(16,), activation="relu",
                                          solver="lbfgs", alpha=0.01, max_iter=2000,
                                          max_fun=30000, random_state=RS),
    }
    try:
        from xgboost import XGBRegressor
        reg["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=6, n_jobs=-1,
                                       random_state=RS, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        reg["LightGBM"] = LGBMRegressor(n_estimators=400, n_jobs=-1, random_state=RS,
                                         verbosity=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        reg["CatBoost"] = CatBoostRegressor(iterations=400, verbose=0, random_seed=RS,
                                             thread_count=-1)
    except Exception:
        pass
    return reg


def build_tuned_regressors() -> Dict[str, Any]:
    """Same models as build_regressors() but wrapped in GridSearchCV for hyperparameter tuning.

    Uses 3-fold inner CV (regular KFold, within each outer training fold) scored on R².
    This eliminates the negative R² caused by default hyperparameters that are poorly
    suited to the battery degradation feature space.
    """
    reg: Dict[str, Any] = {
        "DummyMean":    DummyRegressor(strategy="mean"),
        "DummyMedian":  DummyRegressor(strategy="median"),
        "PreviousRPT":  DummyRegressor(strategy="mean"),
        "Ridge": _gs(
            Ridge(random_state=RS),
            {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
        "Lasso": _gs(
            Lasso(random_state=RS, max_iter=50_000),
            {"alpha": [1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5]},
        ),
        "ElasticNet": _gs(
            ElasticNet(random_state=RS, max_iter=50_000),
            {"alpha": [1e-3, 0.01, 0.1, 0.5], "l1_ratio": [0.2, 0.5, 0.8]},
        ),
        # SVR: C and epsilon are critical — default C=1 was catastrophically off
        "SVR-linear": _gs(
            SVR(kernel="linear"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
             "epsilon": [0.001, 0.01, 0.1]},
        ),
        "SVR-RBF": _gs(
            SVR(kernel="rbf"),
            {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
             "gamma": ["scale", "auto"],
             "epsilon": [0.001, 0.01, 0.1]},
        ),
        # KNN: tune neighbors and weighting scheme
        "KNN": _gs(
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10, 15],
             "weights": ["uniform", "distance"]},
        ),
        # DecisionTree: regularize depth to prevent pure overfit
        "DecisionTree": _gs(
            DecisionTreeRegressor(random_state=RS),
            {"max_depth": [2, 3, 5, 7],
             "min_samples_leaf": [1, 2, 5],
             "min_samples_split": [2, 5, 10]},
        ),
        # Ensembles: tune depth and feature fraction
        "RandomForest": _gs(
            RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=RS),
            {"max_depth": [None, 5, 10],
             "max_features": ["sqrt", 0.5, 0.8],
             "min_samples_leaf": [1, 2, 4]},
        ),
        "ExtraTrees": _gs(
            ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=RS),
            {"max_depth": [None, 5, 10],
             "max_features": ["sqrt", 0.5, 0.8],
             "min_samples_leaf": [1, 2, 4]},
        ),
        # GradientBoosting: depth + learning rate interaction is key
        "GradientBoosting": _gs(
            GradientBoostingRegressor(random_state=RS),
            {"max_depth": [2, 3, 4],
             "learning_rate": [0.05, 0.1, 0.2],
             "n_estimators": [100, 200, 400],
             "subsample": [0.8, 1.0]},
        ),
        # MLP: size and regularization strength
        "MLP": _gs(
            MLPRegressor(activation="relu", solver="adam",
                         max_iter=500, random_state=RS),
            {"hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
             "alpha": [0.001, 0.01, 0.1],
             "learning_rate_init": [0.001, 0.01]},
        ),
    }
    try:
        from xgboost import XGBRegressor
        reg["XGBoost"] = _gs(
            XGBRegressor(n_jobs=1, random_state=RS, verbosity=0),
            {"max_depth": [2, 3, 4],
             "learning_rate": [0.05, 0.1, 0.2],
             "n_estimators": [100, 300, 500],
             "subsample": [0.8, 1.0],
             "colsample_bytree": [0.8, 1.0]},
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        reg["LightGBM"] = _gs(
            LGBMRegressor(n_jobs=1, random_state=RS, verbosity=-1),
            {"max_depth": [3, 5, -1],
             "num_leaves": [15, 31, 63],
             "learning_rate": [0.05, 0.1, 0.2],
             "n_estimators": [100, 200, 400]},
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        reg["CatBoost"] = _gs(
            CatBoostRegressor(verbose=0, random_seed=RS, thread_count=1),
            {"depth": [3, 4, 6],
             "learning_rate": [0.05, 0.1, 0.2],
             "iterations": [200, 400]},
        )
    except Exception:
        pass
    return reg


NEEDS_SCALING = {"Ridge", "Lasso", "ElasticNet", "SVR-linear", "SVR-RBF", "KNN", "MLP"}
