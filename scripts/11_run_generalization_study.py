#!/usr/bin/env python3
"""Leave-one-condition-out and leave-one-factor-level-out generalization study.

Cell-grouped CV answers "a new cell under a known condition". This script
answers the two harder questions the paper actually claims: an unseen operating
*combination* (LOCO) and an unseen factor *level* (extrapolation).

Writes, under ``artifacts/results/generalization/``:

    loco_manifest.parquet              immutable split assignment + split_id
    factor_manifest.parquet
    generalization_predictions.parquet per-row out-of-fold predictions
    generalization_metrics.csv         per-split metrics vs the persistence floor
    generalization_summary.csv         one row per model

Usage::

    python3 scripts/11_run_generalization_study.py
    python3 scripts/11_run_generalization_study.py --studies loco --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import aggregated_regression_metrics  # noqa: E402
from src.io.loaders import load_config  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402
from src.models.degradation_baselines import (  # noqa: E402
    build_degradation_baselines, stress_design_diagnostics,
)
from src.models.regressors import STUDY_RIDGE_ALPHA  # noqa: E402
from src.splits.condition_holdout import (  # noqa: E402
    describe_manifest, iter_splits, leave_one_condition_out,
    leave_one_factor_level_out,
)

TARGET = "delta_next_rpt_Q_Ah"


def _sklearn_models(feature_columns: list[str]):
    """Statistical models, wrapped so they share the baselines' frame API."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def wrap(estimator, scale_target: bool):
        inner = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), estimator,
        )
        return TransformedTargetRegressor(
            regressor=inner, transformer=StandardScaler(),
        ) if scale_target else inner

    return {
        # Frozen penalty, not RidgeCV: see STUDY_RIDGE_ALPHA for why row-level
        # LOO alpha selection leaks across the anchors of a cell.
        "Ridge": (wrap(Ridge(alpha=STUDY_RIDGE_ALPHA), False), feature_columns),
        "ExtraTrees": (wrap(ExtraTreesRegressor(n_estimators=500, random_state=42), False), feature_columns),
        "GradientBoosting": (wrap(GradientBoostingRegressor(random_state=42), False), feature_columns),
        # The neural baseline needs a standardised target: the raw delta has a
        # scale of ~0.007 Ah and the optimiser stalls on it otherwise.
        "MLP": (wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                  max_iter=8000, random_state=42), True), feature_columns),
    }


def _load_frame(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    artifacts = Path(cfg["paths"]["artifacts"])
    unified = artifacts / "features" / "unified.parquet"
    manifest = artifacts / "features" / "unified_feature_manifest.json"
    split_manifest_path = artifacts / "splits" / "split_manifest.parquet"
    if not unified.exists():
        raise SystemExit(
            f"missing {unified}. Run scripts/01_build_features.py first, or point "
            f"SIB_ARTIFACTS at a built artifacts directory."
        )
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    if split_manifest_path.exists():
        split_manifest = pd.read_parquet(split_manifest_path)[["cell", "outer_role"]]
        frame = frame.merge(split_manifest, on="cell", how="left")
        frame = frame[frame["outer_role"].eq("development")].reset_index(drop=True)
        holdout_remaining = int((frame.get("outer_role", pd.Series()) == "holdout").sum())
        assert holdout_remaining == 0, "Holdout cells leaked into development frame"
    features = json.loads(manifest.read_text())["feature_columns"]
    return frame, features


def _evaluate(frame: pd.DataFrame, manifest: pd.DataFrame, study: str,
              feature_columns: list[str]) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    baselines = build_degradation_baselines()
    statistical = _sklearn_models(feature_columns)

    for split_name, train_idx, test_idx, info in iter_splits(manifest, frame):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        y_train = train[TARGET].to_numpy(dtype=float)
        y_test = test[TARGET].to_numpy(dtype=float)

        for name, model in baselines.items():
            fitted = copy.deepcopy(model).fit(train, y_train)
            predictions.append(_rows(test, y_test, fitted.predict(test),
                                     name, split_name, study, info))

        for name, (model, columns) in statistical.items():
            fitted = copy.deepcopy(model)
            try:
                fitted.fit(train[columns], y_train)
                y_hat = fitted.predict(test[columns])
            except Exception as error:  # noqa: BLE001
                print(f"  [warn] {name} failed on {split_name}: {error}")
                continue
            predictions.append(_rows(test, y_test, y_hat, name, split_name,
                                     study, info))

    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def _rows(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
          model: str, split_name: str, study: str, info: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "study": study,
        "split_name": split_name,
        "factor": info["factor"],
        "held_out_value": info["held_out_value"],
        "degenerate": info["degenerate"],
        "split_id": info["split_id"],
        "model": model,
        "anchor_id": test.get("anchor_id", pd.Series(test.index)).to_numpy(),
        "cell": test["cell"].to_numpy(),
        "condition": test["condition"].to_numpy(),
        "y_true": y_true,
        "y_pred": np.asarray(y_pred, dtype=float),
    })


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, split_name, model), group in predictions.groupby(
        ["study", "split_name", "model"], sort=True,
    ):
        error = np.abs(group["y_true"] - group["y_pred"])
        # Note for whoever reads the output: on a LOCO split, cell_macro_MAE
        # equals MAE exactly. That is not a bug -- the test set is one condition
        # and every cell within a condition in this dataset carries an identical
        # anchor count, so the macro average of balanced groups is the pooled
        # average. The distinction only bites when pooling across conditions.
        # The persistence floor for this split: predicting no change at all.
        floor = float(np.abs(group["y_true"]).mean())
        rows.append({
            "study": study,
            "split_name": split_name,
            "model": model,
            "factor": group["factor"].iloc[0],
            "held_out_value": group["held_out_value"].iloc[0],
            "degenerate": bool(group["degenerate"].iloc[0]),
            "n": int(len(group)),
            "n_cells": int(group["cell"].nunique()),
            "MAE": float(error.mean()),
            "cell_macro_MAE": float(error.groupby(group["cell"]).mean().mean()),
            "RMSE": float(np.sqrt((error ** 2).mean())),
            "persistence_MAE": floor,
            "beats_persistence": bool(error.mean() < floor),
        })
    return pd.DataFrame(rows)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", default=["loco", "factor"],
                        choices=["loco", "factor"])
    parser.add_argument("--dry-run", action="store_true",
                        help="build and describe the manifests, fit nothing")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load_frame(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "generalization"
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen] anchors={len(frame)} cells={frame['cell'].nunique()} "
          f"conditions={frame['condition'].nunique()}")
    diagnostics = stress_design_diagnostics(frame)
    print(f"[gen] stress-design condition number "
          f"{diagnostics['condition_number']:.0f}; C-rate exponent identifiable: "
          f"{diagnostics['c_rate_exponent_identifiable']}")
    (out / "stress_design_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str) + "\n", encoding="utf-8",
    )

    manifests: dict[str, pd.DataFrame] = {}
    if "loco" in args.studies:
        manifests["loco"] = leave_one_condition_out(frame)
    if "factor" in args.studies:
        manifests["factor"] = leave_one_factor_level_out(frame)

    all_predictions, all_metrics = [], []
    for study, manifest in manifests.items():
        manifest.to_parquet(out / f"{study}_manifest.parquet", index=False)
        summary = describe_manifest(manifest)
        summary.to_csv(out / f"{study}_manifest_summary.csv", index=False)
        n_degenerate = int(summary["degenerate"].sum())
        print(f"[gen] {study}: {manifest['split_name'].nunique()} splits "
              f"({n_degenerate} degenerate, i.e. a single held-out cell)")
        if args.dry_run:
            print(summary.to_string(index=False))
            continue
        predictions = _evaluate(frame, manifest, study, feature_columns)
        if predictions.empty:
            continue
        all_predictions.append(predictions)
        all_metrics.append(_metrics(predictions))

    if args.dry_run:
        print("[gen] dry run: manifests written, no models fitted")
        return 0
    if not all_predictions:
        print("[gen] no usable splits")
        return 1

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions.to_parquet(out / "generalization_predictions.parquet", index=False)
    metrics.to_csv(out / "generalization_metrics.csv", index=False)

    # Headline: mean over splits, and how often each model beats doing nothing.
    # Degenerate splits are reported separately -- a single-anchor test set is
    # too noisy to average into a headline.
    solid = metrics[~metrics["degenerate"]]
    summary = (
        solid.groupby(["study", "model"])
        .agg(splits=("split_name", "nunique"),
             mean_MAE=("MAE", "mean"),
             mean_cell_macro_MAE=("cell_macro_MAE", "mean"),
             mean_persistence_MAE=("persistence_MAE", "mean"),
             splits_beating_persistence=("beats_persistence", "sum"))
        .reset_index()
        .sort_values(["study", "mean_cell_macro_MAE"])
    )
    summary.to_csv(out / "generalization_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\n[gen] wrote {out}")
    return 0


def main() -> int:
    """Run under a capture log so a finished run is still debuggable.

    Everything these studies printed used to go to stdout and nowhere else.
    """
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "generalization" / "logs"
    with run_log("11_run_generalization_study", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
