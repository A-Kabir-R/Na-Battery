"""Per-iteration train/val loss capture for models that expose an iterative fit.

Outer train-vs-test scalar metrics come from `run_experiment.run_one`. This
module handles diagnostic curves on an inner training-cell validation split:

  - XGBoost (reg + clf)         via eval_set + evals_result()
  - LightGBM (reg + clf)        via eval_set + evals_result_
  - CatBoost (reg + clf)        via eval_set + get_evals_result()
  - sklearn GradientBoosting    via train_score_ + staged_predict on X_val
  - sklearn MLP (reg + clf)     via loss_curve_  (train only — no val curve
                                  without giving up part of X_tr)

For every other classical model there is no meaningful per-iteration curve
(one-shot fit), and `fit_with_curves` returns `(est, None)`.

The fitted estimator returned from `fit_with_curves` is a normal fitted sklearn-
style estimator. It is disposable; the outer estimator is fitted separately on
all outer-training cells and never receives outer-test data.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

_XGB_NAMES  = {"XGBoost", "XGBoost-clf"}
_LGBM_NAMES = {"LightGBM", "LightGBM-clf"}
_CAT_NAMES  = {"CatBoost", "CatBoost-clf"}
_SKGB_NAMES = {"GradientBoosting"}                      # sklearn only exposes reg here
_MLP_NAMES  = {"MLP", "MLP-clf"}

CURVE_MODELS = _XGB_NAMES | _LGBM_NAMES | _CAT_NAMES | _SKGB_NAMES | _MLP_NAMES


def _curve_df(train: np.ndarray, val: Optional[np.ndarray], loss_name: str) -> pd.DataFrame:
    n = len(train)
    rows = [{"iteration": i + 1, "split": "train",
             "loss_name": loss_name, "loss": float(train[i])} for i in range(n)]
    if val is not None:
        m = len(val)
        for i in range(m):
            rows.append({"iteration": i + 1, "split": "val",
                         "loss_name": loss_name, "loss": float(val[i])})
    return pd.DataFrame(rows)


def _fit_xgb(est, X_tr, y_tr, X_te, y_te, kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    eval_metric = "rmse" if kind == "regression" else "logloss"
    try:
        est.set_params(eval_metric=eval_metric)
    except Exception:
        pass
    try:
        est.fit(X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_te, y_te)],
                verbose=False)
        res = est.evals_result()
        tr = np.asarray(res["validation_0"][eval_metric], dtype=float)
        va = np.asarray(res["validation_1"][eval_metric], dtype=float)
        return est, _curve_df(tr, va, eval_metric)
    except Exception:
        est.fit(X_tr, y_tr)
        return est, None


def _fit_lgbm(est, X_tr, y_tr, X_te, y_te, kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    eval_metric = "rmse" if kind == "regression" else "binary_logloss"
    try:
        import lightgbm as lgb
        record: dict = {}
        try:
            est.fit(X_tr, y_tr,
                    eval_X=[X_tr, X_te],
                    eval_y=[y_tr, y_te],
                    eval_names=["train", "val"],
                    eval_metric=eval_metric,
                    callbacks=[lgb.record_evaluation(record)])
        except TypeError:
            est.fit(X_tr, y_tr,
                    eval_set=[(X_tr, y_tr), (X_te, y_te)],
                    eval_names=["train", "val"],
                    eval_metric=eval_metric,
                    callbacks=[lgb.record_evaluation(record)])
        tr = np.asarray(record["train"][eval_metric], dtype=float)
        va = np.asarray(record["val"][eval_metric], dtype=float)
        return est, _curve_df(tr, va, eval_metric)
    except Exception:
        est.fit(X_tr, y_tr)
        return est, None


def _fit_catboost(est, X_tr, y_tr, X_te, y_te, kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    try:
        est.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
        res = est.get_evals_result()
        # CatBoost keys are e.g. 'learn' / 'validation' with the configured loss name.
        learn = res.get("learn", {})
        valid = res.get("validation", {})
        if not learn:
            return est, None
        loss_name = next(iter(learn.keys()))
        tr = np.asarray(learn[loss_name], dtype=float)
        va = np.asarray(valid.get(loss_name, []), dtype=float) if valid else None
        return est, _curve_df(tr, va if (va is not None and len(va)) else None, loss_name)
    except Exception:
        est.fit(X_tr, y_tr)
        return est, None


def _fit_sk_gb(est, X_tr, y_tr, X_te, y_te, kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    try:
        est.fit(X_tr, y_tr)
        # sklearn GBR: train_score_ is the loss on X_tr at each stage.
        tr = np.asarray(getattr(est, "train_score_", []), dtype=float)
        # val loss: MSE per stage on X_te via staged_predict.
        try:
            va_list = [mean_squared_error(y_te, y_hat)
                       for y_hat in est.staged_predict(X_te)]
            va = np.asarray(va_list, dtype=float)
            loss_name = "MSE"
            # tr from sklearn is deviance/MSE depending on loss; align by MSE on X_tr too
            # so both curves are on the same scale.
            tr_list = [mean_squared_error(y_tr, y_hat)
                       for y_hat in est.staged_predict(X_tr)]
            tr = np.asarray(tr_list, dtype=float)
        except Exception:
            va = None
            loss_name = "train_score"
        if len(tr) == 0:
            return est, None
        return est, _curve_df(tr, va, loss_name)
    except Exception:
        est.fit(X_tr, y_tr)
        return est, None


def _fit_mlp(est, X_tr, y_tr, X_te, y_te, kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    try:
        est.fit(X_tr, y_tr)
        tr = np.asarray(getattr(est, "loss_curve_", []), dtype=float)
        if len(tr) == 0:
            return est, None
        # sklearn MLP does not track a separate val curve unless early_stopping=True
        # (which reserves an internal split). We intentionally don't enable that.
        return est, _curve_df(tr, None, "train_loss")
    except Exception:
        est.fit(X_tr, y_tr)
        return est, None


def fit_with_curves(name: str, est, X_tr, y_tr, X_te, y_te,
                    kind: str) -> Tuple[Any, Optional[pd.DataFrame]]:
    """Fit `est` and return (fitted_est, per-iteration loss dataframe or None).

    The dataframe schema (when present) is:
        iteration:int, split:{'train','val'}, loss_name:str, loss:float
    """
    if name in _XGB_NAMES:
        return _fit_xgb(est, X_tr, y_tr, X_te, y_te, kind)
    if name in _LGBM_NAMES:
        return _fit_lgbm(est, X_tr, y_tr, X_te, y_te, kind)
    if name in _CAT_NAMES:
        return _fit_catboost(est, X_tr, y_tr, X_te, y_te, kind)
    if name in _SKGB_NAMES:
        return _fit_sk_gb(est, X_tr, y_tr, X_te, y_te, kind)
    if name in _MLP_NAMES:
        return _fit_mlp(est, X_tr, y_tr, X_te, y_te, kind)
    est.fit(X_tr, y_tr)
    return est, None
