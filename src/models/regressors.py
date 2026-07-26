"""Classical regressor registry."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

RS = 42


def _try(name: str, factory):
    try:
        return factory()
    except Exception:
        return None


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
        "MLP":              MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=RS),
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


NEEDS_SCALING = {"Ridge", "Lasso", "ElasticNet", "SVR-linear", "SVR-RBF", "KNN", "MLP"}
