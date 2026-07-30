"""Leakage-safe grouped outer-fold experiment runner."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..evaluation.loss_curves import CURVE_MODELS, fit_with_curves
from ..evaluation.metrics import classification_metrics, regression_metrics
from ..features.selection import FoldPreprocessor
from ..io.loaders import load_config
from ..models.registry import needs_scaling
from ..preprocessing.fold_transform import UnifiedFoldPreprocessor
from ..preprocessing.unified_feature_registry import (
    REGISTRY_VERSION, registry_hash,
)
from ..splits.group_kfold import (
    assert_no_cell_overlap,
    build_locked_split_manifest,
    make_folds,
    validate_locked_split_manifest,
)

# Unsafe legacy SOH/RUL/proxy-failure benchmarks are intentionally disabled.
REGRESSION_TARGETS = (
    "next_rpt_Q_Ah",
    "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah",
    "delta_next_rpt_SOH_pct",
)
CLASSIFICATION_TARGETS: tuple[str, ...] = ()
PIPELINE_VERSION = "unified-registry-v4"

#: The one preprocessing level whose feature matrix comes from the fixed
#: registry. For this level Ridge and ExtraTrees use exactly the columns and
#: exactly the fold-local transform that NaPINN-Q uses.
UNIFIED_PREPROCESSING = "unified"


def _predict_score(est, features: np.ndarray, kind: str):
    if kind != "classification" or not hasattr(est, "predict_proba"):
        return None
    try:
        return est.predict_proba(features)[:, 1]
    except Exception:
        return None


def _metrics(y_true, y_pred, y_score, kind: str, target: str) -> Dict[str, float]:
    if kind == "regression":
        metrics = regression_metrics(y_true, y_pred)
        if target.startswith("delta_"):
            metrics.pop("MAPE", None)
        return metrics
    return classification_metrics(y_true, y_pred, y_score)


def _curve_dir() -> Path:
    directory = Path(load_config()["paths"]["artifacts"]) / "loss_curves"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def _fold_dir(preprocessing: str, target: str, model_name: str, fold: int) -> Path:
    root = Path(load_config()["paths"]["artifacts"]) / "folds"
    return root / _safe_name(preprocessing) / _safe_name(target) / _safe_name(model_name) / f"fold_{fold}"


def _holdout_dir(preprocessing: str, target: str, model_name: str) -> Path:
    root = Path(load_config()["paths"]["artifacts"]) / "folds"
    return root / _safe_name(preprocessing) / _safe_name(target) / _safe_name(model_name) / "holdout"


def _experiment_fingerprint(data: pd.DataFrame, preprocessing: str, target: str,
                            model_name: str, model, kind: str,
                            n_splits: int, random_state: int) -> str:
    data_hash = sha256(
        pd.util.hash_pandas_object(data, index=True, categorize=True).to_numpy().tobytes()
    ).hexdigest()
    cfg = load_config()
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "data_hash": data_hash,
        "preprocessing": preprocessing,
        "target": target,
        "model_name": model_name,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_params": model.get_params(deep=True),
        "kind": kind,
        "n_splits": n_splits,
        "random_state": random_state,
        "config": {
            key: cfg.get(key) for key in
            ["cleaning", "protocol", "preprocessing", "targets", "split", "evaluation"]
        },
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return sha256(serialized).hexdigest()


def _fold_fingerprint(experiment_fingerprint: str, train_cells: list[str],
                      test_cells: list[str]) -> str:
    payload = json.dumps({
        "experiment": experiment_fingerprint,
        "train_cells": train_cells,
        "test_cells": test_cells,
    }, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _prediction_frame(data: pd.DataFrame, indices: np.ndarray, split: str,
                      target: str, predictions: np.ndarray,
                      scores: np.ndarray | None, *, preprocessing: str,
                      model_name: str, fold: int) -> pd.DataFrame:
    identity_columns = [
        column for column in [
            "anchor_id", "condition", "cell", "file_id", "path", "visit", "global_cycle",
            "initial_rpt_Q_Ah", "previous_rpt_Q_Ah", "previous_rpt_SOH_pct",
            "next_rpt_visit", "next_rpt_match_method", "next_rpt_horizon_days",
        ]
        if column in data.columns
    ]
    frame = data.iloc[indices][identity_columns].reset_index().rename(columns={"index": "row_index"})
    frame["split"] = split
    frame["evaluation_status"] = (
        "exploratory_locked_holdout_benchmark" if split == "holdout"
        else "development_only"
    )
    frame["preprocessing"] = preprocessing
    frame["target"] = target
    frame["model"] = model_name
    frame["fold"] = fold
    frame["y_true"] = data.iloc[indices][target].to_numpy()
    frame["y_pred"] = predictions
    if scores is not None:
        frame["y_score"] = scores
    return frame


def _prediction_plausibility(predictions: np.ndarray, target: str) -> tuple[np.ndarray, list[str]]:
    bounds = load_config().get("evaluation", {}).get("prediction_bounds", {}).get(target)
    finite = np.isfinite(predictions)
    valid = finite.copy()
    flags = np.full(len(predictions), "", dtype=object)
    flags[~finite] = "nonfinite_prediction"
    if bounds is not None:
        low, high = float(bounds[0]), float(bounds[1])
        out_of_range = finite & ((predictions < low) | (predictions > high))
        valid &= ~out_of_range
        flags[out_of_range] = "prediction_out_of_bounds"
    return valid, flags.tolist()


def _persistence_predictions(train: pd.DataFrame, test: pd.DataFrame,
                             target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target in {"delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct"}:
        return np.zeros(len(train)), np.zeros(len(test)), ["zero_change"]
    source = {
        "next_rpt_Q_Ah": "previous_rpt_Q_Ah",
        "next_rpt_SOH_pct": "previous_rpt_SOH_pct",
    }.get(target)
    if source is None or source not in train.columns:
        raise ValueError(f"PreviousRPT baseline is unsupported for target {target}")
    return (
        train[source].to_numpy(dtype=float),
        test[source].to_numpy(dtype=float),
        [source],
    )


def _build_preprocessor(preprocessing: str, model_name: str, kind: str):
    """Return the fold-local transformer for this run.

    For the ``unified`` level every model shares one
    :class:`UnifiedFoldPreprocessor` with ``scale=True``, so Ridge and
    ExtraTrees receive a byte-identical matrix to NaPINN-Q's auxiliary tensor
    (plus the stress coordinate and initial state, which the PINN consumes
    structurally instead). ExtraTrees is scale invariant, so forcing the same
    scaling costs nothing and removes a source of unfair difference.
    """
    if preprocessing == UNIFIED_PREPROCESSING:
        return UnifiedFoldPreprocessor(scale=True)
    return FoldPreprocessor(scale=needs_scaling(model_name, kind))


def _fit_partition(train: pd.DataFrame, test: pd.DataFrame, *, target: str,
                   model_name: str, model, kind: str,
                   preprocessing: str = ""):
    y_train = train[target].to_numpy()
    if model_name == "PreviousRPT":
        train_prediction, test_prediction, features = _persistence_predictions(train, test, target)
        return None, {"baseline": "PreviousRPT", "features": features}, features, train_prediction, test_prediction, None, None
    preprocessor = _build_preprocessor(preprocessing, model_name, kind)
    if isinstance(preprocessor, UnifiedFoldPreprocessor):
        x_train = preprocessor.fit_transform(train)
    else:
        x_train = preprocessor.fit_transform(train, target=target)
    x_test = preprocessor.transform(test)
    estimator = clone(model)
    estimator.fit(x_train, y_train)
    train_prediction = estimator.predict(x_train)
    test_prediction = estimator.predict(x_test)
    return (
        preprocessor,
        estimator,
        list(preprocessor.feature_names_ or []),
        train_prediction,
        test_prediction,
        _predict_score(estimator, x_train, kind),
        _predict_score(estimator, x_test, kind),
    )


def _model_importance(estimator: Any, features: list[str]) -> pd.DataFrame:
    if not features or estimator is None or isinstance(estimator, dict):
        return pd.DataFrame()
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
        importance_type = "absolute_coefficient"
    else:
        return pd.DataFrame()
    if len(values) != len(features):
        return pd.DataFrame()
    denominator = float(np.nansum(np.abs(values)))
    normalized = np.abs(values) / denominator if denominator > 0 else np.zeros(len(values))
    return pd.DataFrame({
        "feature": features,
        "importance": values,
        "normalized_importance": normalized,
        "importance_type": importance_type,
    })


def _diagnostic_curve(data: pd.DataFrame, train_indices: np.ndarray, target: str,
                      model_name: str, model, kind: str,
                      random_state: int,
                      preprocessing: str = "") -> pd.DataFrame | None:
    """Fit a disposable inner-fold model for loss curves only."""
    if model_name not in CURVE_MODELS:
        return None
    outer_train = data.iloc[train_indices].reset_index(drop=True)
    group_count = outer_train["cell"].nunique()
    if group_count < 2:
        return None
    inner_splits = min(3, group_count)
    inner_train, inner_val = next(iter(make_folds(
        outer_train,
        n_splits=inner_splits,
        random_state=random_state,
    )))
    assert_no_cell_overlap(inner_train, inner_val, outer_train["cell"])
    preprocessor = _build_preprocessor(preprocessing, model_name, kind)
    if isinstance(preprocessor, UnifiedFoldPreprocessor):
        x_inner_train = preprocessor.fit_transform(outer_train.iloc[inner_train])
    else:
        x_inner_train = preprocessor.fit_transform(
            outer_train.iloc[inner_train], target=target,
        )
    x_inner_val = preprocessor.transform(outer_train.iloc[inner_val])
    y = outer_train[target].to_numpy()
    _, curve = fit_with_curves(
        model_name,
        clone(model),
        x_inner_train,
        y[inner_train],
        x_inner_val,
        y[inner_val],
        kind,
    )
    return curve


def _metric_rows(*, preprocessing: str, target: str, model_name: str, fold: int,
                 split: str, metrics: dict[str, float], train: pd.DataFrame,
                 test: pd.DataFrame, n_features: int, fingerprint: str,
                 invalid_predictions: int) -> list[dict[str, Any]]:
    scientific_status = "valid" if invalid_predictions == 0 else "invalid_predictions"
    return [{
        "preprocessing": preprocessing,
        "target": target,
        "model": model_name,
        "fold": fold,
        "split": split,
        "evaluation_status": (
            "exploratory_locked_holdout_benchmark" if split == "holdout"
            else "development_only"
        ),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_cells": int(train["cell"].nunique()),
        "n_test_cells": int(test["cell"].nunique()),
        "n_features": int(n_features),
        "invalid_prediction_count": int(invalid_predictions),
        "scientific_status": scientific_status,
        "metric": metric,
        "value": float(value),
        "status": "completed",
        "error_message": None,
        "fingerprint": fingerprint,
    } for metric, value in metrics.items()]


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_provenance_artifacts(directory: Path, preprocessor: Any,
                                features: list[str], *, preprocessing: str,
                                target: str, model_name: str, fold: int,
                                n_splits: int, random_state: int) -> None:
    """Write the per-fold provenance files the revision requires.

    Emits ``feature_columns.json``, ``feature_hash.json``,
    ``preprocessor_state.json`` and ``config_snapshot.json`` so a classical fold
    can be compared column-for-column against the corresponding PINN fold.
    """
    _atomic_json({
        "feature_columns": list(features),
        "n_features": len(features),
        "registry_version": REGISTRY_VERSION,
    }, directory / "feature_columns.json")

    feature_hash = (
        preprocessor.feature_hash()
        if hasattr(preprocessor, "feature_hash") else None
    )
    _atomic_json({
        "feature_hash": feature_hash,
        "registry_hash": registry_hash(),
        "preprocessor_state_hash": (
            preprocessor.state_hash()
            if hasattr(preprocessor, "state_hash") else None
        ),
    }, directory / "feature_hash.json")

    if hasattr(preprocessor, "state_dict"):
        _atomic_json(preprocessor.state_dict(), directory / "preprocessor_state.json")

    cfg = load_config()
    _atomic_json({
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing": preprocessing,
        "target": target,
        "model": model_name,
        "fold": fold,
        "n_splits": n_splits,
        "random_state": random_state,
        "git_commit": _git_commit(),
        "config": {
            key: cfg.get(key)
            for key in ("cleaning", "protocol", "preprocessing", "targets",
                        "split", "evaluation", "experiment", "models")
        },
    }, directory / "config_snapshot.json")


def _completed_rows(directory: Path, fingerprint: str) -> list[dict[str, Any]] | None:
    metrics_path = directory / "fold_metrics.parquet"
    status_path = directory / "fold_status.json"
    required = [
        metrics_path, status_path, directory / "predictions.parquet",
        directory / "model.joblib", directory / "preprocessor.joblib",
        directory / "selected_features.json",
    ]
    if not all(path.exists() for path in required):
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed" and status.get("fingerprint") == fingerprint:
            return pd.read_parquet(metrics_path).to_dict(orient="records")
    except Exception:
        return None
    return None


def _invalidate_partition(directory: Path) -> None:
    """Remove known outputs before a rerun so failures cannot expose stale files."""
    for filename in (
        "fold_metrics.parquet", "fold_status.json", "predictions.parquet",
        "model.joblib", "preprocessor.joblib", "selected_features.json",
        "feature_importance.parquet",
    ):
        (directory / filename).unlink(missing_ok=True)


def run_one(df: pd.DataFrame, preprocessing: str, target: str, model_name: str,
            model, kind: str, n_splits: int = 5, random_state: int = 42,
            split_manifest: pd.DataFrame | None = None,
            evaluate_holdout: bool | None = None) -> List[Dict[str, Any]]:
    """Run cross-validation and, only when explicitly enabled, the holdout.

    ``evaluate_holdout`` defaults to ``experiment.evaluate_holdout`` in the
    config, which ships as ``false``. With it false the locked holdout partition
    is never read, so development work cannot leak information from it. Set it
    true exactly once, after every setting is frozen.
    """
    if target not in df.columns:
        return []
    data = df.dropna(subset=[target]).reset_index(drop=True)
    if data.empty:
        return []
    if split_manifest is None:
        cfg = load_config()["split"]
        split_manifest = build_locked_split_manifest(
            data,
            holdout_fraction=float(cfg.get("holdout_fraction", 0.25)),
            n_splits=n_splits,
            random_state=random_state,
            holdout_candidates=int(cfg.get("holdout_candidates", 1024)),
        )
    validate_locked_split_manifest(
        data, split_manifest, n_splits=n_splits,
        holdout_fraction=float(load_config()["split"].get("holdout_fraction", 0.25)),
        random_state=random_state,
    )
    assignments = split_manifest[["cell", "outer_role", "cv_fold"]].copy()
    assignments["cell"] = assignments["cell"].astype(str)
    data["cell"] = data["cell"].astype(str)
    data = data.merge(assignments, on="cell", how="left", validate="many_to_one")
    if data["outer_role"].isna().any():
        raise ValueError("target rows contain cells missing from the split manifest")

    rows: List[Dict[str, Any]] = []
    curve_dir = _curve_dir()
    experiment_fingerprint = _experiment_fingerprint(
        data, preprocessing, target, model_name, model, kind, n_splits, random_state
    )
    development_mask = data["outer_role"].eq("development")
    holdout_mask = data["outer_role"].eq("holdout")
    if not development_mask.any() or not holdout_mask.any():
        raise ValueError("locked split must contain target rows in development and holdout")

    for fold in range(n_splits):
        validation_mask = development_mask & data["cv_fold"].eq(fold)
        train_mask = development_mask & ~data["cv_fold"].eq(fold)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(validation_mask.to_numpy())
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"CV fold {fold} has no target rows")
        assert_no_cell_overlap(train_indices, validation_indices, data["cell"])
        train_cells = sorted(data.iloc[train_indices]["cell"].unique())
        validation_cells = sorted(data.iloc[validation_indices]["cell"].unique())
        fold_fingerprint = _fold_fingerprint(
            experiment_fingerprint, train_cells, validation_cells
        )
        fold_dir = _fold_dir(preprocessing, target, model_name, fold)
        reused = _completed_rows(fold_dir, fold_fingerprint)
        if reused is not None:
            rows.extend(reused)
            continue

        _invalidate_partition(fold_dir)
        status_path = fold_dir / "fold_status.json"
        _atomic_json({
            "status": "running", "fold": fold, "evaluation_role": "development_cv",
            "fingerprint": fold_fingerprint,
        }, status_path)
        error_message: str | None = None
        fold_rows: list[dict[str, Any]] = []
        try:
            train = data.iloc[train_indices]
            validation = data.iloc[validation_indices]
            curve = _diagnostic_curve(
                data, train_indices, target, model_name, model, kind,
                random_state + fold, preprocessing=preprocessing,
            )
            if curve is not None and not curve.empty:
                curve_output = curve.assign(
                    preprocessing=preprocessing, target=target, model=model_name,
                    fold=fold, validation_role="development_training_cells",
                )
                filename = f"{preprocessing}__{target}__{model_name}__fold{fold}.parquet"
                _atomic_parquet(curve_output, curve_dir / filename)

            (preprocessor, estimator, features, train_prediction, validation_prediction,
             train_score, validation_score) = _fit_partition(
                train, validation, target=target, model_name=model_name,
                model=model, kind=kind, preprocessing=preprocessing,
            )
            train_valid, train_flags = _prediction_plausibility(train_prediction, target)
            validation_valid, validation_flags = _prediction_plausibility(
                validation_prediction, target
            )
            fold_rows.extend(_metric_rows(
                preprocessing=preprocessing, target=target, model_name=model_name, fold=fold,
                split="cv_train", metrics=_metrics(
                    train[target].to_numpy(), train_prediction, train_score, kind, target
                ), train=train, test=validation, n_features=len(features),
                fingerprint=fold_fingerprint,
                invalid_predictions=int((~train_valid).sum()),
            ))
            fold_rows.extend(_metric_rows(
                preprocessing=preprocessing, target=target, model_name=model_name, fold=fold,
                split="cv_validation", metrics=_metrics(
                    validation[target].to_numpy(), validation_prediction, validation_score, kind, target
                ), train=train, test=validation, n_features=len(features),
                fingerprint=fold_fingerprint,
                invalid_predictions=int((~validation_valid).sum()),
            ))
            train_frame = _prediction_frame(
                data, train_indices, "cv_train", target, train_prediction, train_score,
                preprocessing=preprocessing, model_name=model_name, fold=fold,
            )
            validation_frame = _prediction_frame(
                data, validation_indices, "cv_validation", target,
                validation_prediction, validation_score, preprocessing=preprocessing,
                model_name=model_name, fold=fold,
            )
            train_frame["prediction_valid"] = train_valid
            train_frame["prediction_qc_flags"] = train_flags
            validation_frame["prediction_valid"] = validation_valid
            validation_frame["prediction_qc_flags"] = validation_flags
            _atomic_parquet(
                pd.concat([train_frame, validation_frame], ignore_index=True),
                fold_dir / "predictions.parquet",
            )
            _atomic_joblib(estimator, fold_dir / "model.joblib")
            _atomic_joblib(preprocessor, fold_dir / "preprocessor.joblib")
            importance = _model_importance(estimator, features)
            if not importance.empty:
                _atomic_parquet(
                    importance.assign(
                        preprocessing=preprocessing, target=target, model=model_name,
                        fold=fold, evaluation_role="development_cv",
                    ),
                    fold_dir / "feature_importance.parquet",
                )
            _atomic_json({
                "features": features, "target": target,
                "validation_cells": validation_cells, "fingerprint": fold_fingerprint,
            }, fold_dir / "selected_features.json")
            _write_provenance_artifacts(
                fold_dir, preprocessor, features, preprocessing=preprocessing,
                target=target, model_name=model_name, fold=fold,
                n_splits=n_splits, random_state=random_state,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            fold_rows.append({
                "preprocessing": preprocessing, "target": target, "model": model_name,
                "fold": fold, "split": "cv_validation",
                "evaluation_status": "development_only", "n_train": len(train_indices),
                "n_test": len(validation_indices),
                "n_train_cells": data.iloc[train_indices]["cell"].nunique(),
                "n_test_cells": data.iloc[validation_indices]["cell"].nunique(),
                "n_features": 0, "invalid_prediction_count": 0,
                "scientific_status": "fit_failed", "metric": "fit_error", "value": np.nan,
                "status": "failed", "error_message": error_message,
                "fingerprint": fold_fingerprint,
            })
        _atomic_parquet(pd.DataFrame(fold_rows), fold_dir / "fold_metrics.parquet")
        _atomic_json({
            "status": "failed" if error_message else "completed", "fold": fold,
            "evaluation_role": "development_cv", "error_message": error_message,
            "fingerprint": fold_fingerprint,
        }, status_path)
        rows.extend(fold_rows)

    if evaluate_holdout is None:
        evaluate_holdout = bool(
            load_config().get("experiment", {}).get("evaluate_holdout", False)
        )
    if not evaluate_holdout:
        # Development-only mode: the holdout partition is not read at all.
        return rows

    development_indices = np.flatnonzero(development_mask.to_numpy())
    holdout_indices = np.flatnonzero(holdout_mask.to_numpy())
    assert_no_cell_overlap(development_indices, holdout_indices, data["cell"])
    holdout_fingerprint = _fold_fingerprint(
        experiment_fingerprint,
        sorted(data.iloc[development_indices]["cell"].unique()),
        sorted(data.iloc[holdout_indices]["cell"].unique()),
    )
    final_dir = _holdout_dir(preprocessing, target, model_name)
    reused = _completed_rows(final_dir, holdout_fingerprint)
    if reused is not None:
        rows.extend(reused)
        return rows

    _invalidate_partition(final_dir)
    status_path = final_dir / "fold_status.json"
    _atomic_json({
        "status": "running", "fold": -1, "evaluation_role": "locked_holdout",
        "fingerprint": holdout_fingerprint,
    }, status_path)
    final_rows: list[dict[str, Any]] = []
    error_message = None
    try:
        development = data.iloc[development_indices]
        holdout = data.iloc[holdout_indices]
        (preprocessor, estimator, features, development_prediction, holdout_prediction,
         development_score, holdout_score) = _fit_partition(
            development, holdout, target=target, model_name=model_name,
            model=model, kind=kind, preprocessing=preprocessing,
        )
        development_valid, development_flags = _prediction_plausibility(
            development_prediction, target
        )
        holdout_valid, holdout_flags = _prediction_plausibility(holdout_prediction, target)
        final_rows.extend(_metric_rows(
            preprocessing=preprocessing, target=target, model_name=model_name, fold=-1,
            split="development_fit", metrics=_metrics(
                development[target].to_numpy(), development_prediction, development_score, kind, target
            ), train=development, test=holdout, n_features=len(features),
            fingerprint=holdout_fingerprint,
            invalid_predictions=int((~development_valid).sum()),
        ))
        final_rows.extend(_metric_rows(
            preprocessing=preprocessing, target=target, model_name=model_name, fold=-1,
            split="holdout", metrics=_metrics(
                holdout[target].to_numpy(), holdout_prediction, holdout_score, kind, target
            ), train=development, test=holdout, n_features=len(features),
            fingerprint=holdout_fingerprint,
            invalid_predictions=int((~holdout_valid).sum()),
        ))
        development_frame = _prediction_frame(
            data, development_indices, "development_fit", target,
            development_prediction, development_score, preprocessing=preprocessing,
            model_name=model_name, fold=-1,
        )
        holdout_frame = _prediction_frame(
            data, holdout_indices, "holdout", target, holdout_prediction,
            holdout_score, preprocessing=preprocessing, model_name=model_name, fold=-1,
        )
        development_frame["prediction_valid"] = development_valid
        development_frame["prediction_qc_flags"] = development_flags
        holdout_frame["prediction_valid"] = holdout_valid
        holdout_frame["prediction_qc_flags"] = holdout_flags
        _atomic_parquet(
            pd.concat([development_frame, holdout_frame], ignore_index=True),
            final_dir / "predictions.parquet",
        )
        _atomic_joblib(estimator, final_dir / "model.joblib")
        _atomic_joblib(preprocessor, final_dir / "preprocessor.joblib")
        importance = _model_importance(estimator, features)
        if not importance.empty:
            _atomic_parquet(
                importance.assign(
                    preprocessing=preprocessing, target=target, model=model_name,
                    fold=-1, evaluation_role="holdout_final",
                ),
                final_dir / "feature_importance.parquet",
            )
        _atomic_json({
            "features": features, "target": target,
            "holdout_cells": sorted(holdout["cell"].unique()),
            "fingerprint": holdout_fingerprint,
        }, final_dir / "selected_features.json")
        _write_provenance_artifacts(
            final_dir, preprocessor, features, preprocessing=preprocessing,
            target=target, model_name=model_name, fold=-1,
            n_splits=n_splits, random_state=random_state,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        final_rows.append({
            "preprocessing": preprocessing, "target": target, "model": model_name,
            "fold": -1, "split": "holdout",
            "evaluation_status": "exploratory_locked_holdout_benchmark",
            "n_train": len(development_indices),
            "n_test": len(holdout_indices),
            "n_train_cells": data.iloc[development_indices]["cell"].nunique(),
            "n_test_cells": data.iloc[holdout_indices]["cell"].nunique(),
            "n_features": 0, "invalid_prediction_count": 0,
            "scientific_status": "fit_failed", "metric": "fit_error", "value": np.nan,
            "status": "failed", "error_message": error_message,
            "fingerprint": holdout_fingerprint,
        })
    _atomic_parquet(pd.DataFrame(final_rows), final_dir / "fold_metrics.parquet")
    _atomic_json({
        "status": "failed" if error_message else "completed", "fold": -1,
        "evaluation_role": "locked_holdout", "error_message": error_message,
        "fingerprint": holdout_fingerprint,
    }, status_path)
    rows.extend(final_rows)
    return rows
