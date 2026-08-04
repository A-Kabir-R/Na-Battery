#!/usr/bin/env python3
"""Cell-clustered paired bootstrap: NaPINN-Q vs classical models.

Separate per-model confidence intervals are insufficient to claim NaPINN-Q beats
a classical model because they do not account for the within-cell correlation
shared by all models on the same anchors. This script computes the correct
paired test: each anchor contributes one signed difference, and full cells are
resampled to preserve intra-cell correlation.

Protocol
--------
- Primary comparison: seed 42 (preregistered headline seed).
- Comparators: Ridge, ExtraTrees, GradientBoosting, MLP (any model with
  per-anchor OOF predictions available).
- Per-anchor errors are obtained by running a 5-fold development CV fresh (no
  holdout data touched). The PINN per-anchor predictions are loaded from the
  existing outer-val predictions.parquet files (also OOF).
- Bootstrap resamples *complete cells* with replacement so that the within-cell
  autocorrelation structure is preserved.
- 10 000 bootstrap draws; 95% confidence interval.

Writes, under ``artifacts/results/bootstrap/``:

    paired_bootstrap_seed42.csv     primary comparison table
    paired_bootstrap_sensitivity.csv  all seeds (43-46 as sensitivity)

Usage::

    python3 scripts/16_run_paired_bootstrap.py
    python3 scripts/16_run_paired_bootstrap.py --reference ExtraTrees --n-bootstrap 20000
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

from src.evaluation.bootstrap import paired_cell_bootstrap  # noqa: E402
from src.io.loaders import load_config  # noqa: E402
from src.pinn.inference import load_fold_predictions  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _classical_models(feature_columns: list[str]):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.models.regressors import STUDY_RIDGE_ALPHA

    def wrap(estimator, scale_target: bool = False):
        inner = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), estimator,
        )
        if not scale_target:
            return inner
        return TransformedTargetRegressor(
            regressor=inner, transformer=StandardScaler(),
        )

    return {
        "Ridge": wrap(Ridge(alpha=STUDY_RIDGE_ALPHA)),
        "ExtraTrees": wrap(ExtraTreesRegressor(n_estimators=400, random_state=42)),
        "GradientBoosting": wrap(GradientBoostingRegressor(random_state=42)),
        "MLP": wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                 max_iter=8000, random_state=42), scale_target=True),
    }


def _load(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    artifacts = Path(cfg["paths"]["artifacts"])
    unified = artifacts / "features" / "unified.parquet"
    manifest = artifacts / "features" / "unified_feature_manifest.json"
    split_manifest_path = artifacts / "splits" / "split_manifest.parquet"
    if not unified.exists():
        raise SystemExit(f"missing {unified}. Run scripts/01_build_features.py first.")
    if not split_manifest_path.exists():
        raise SystemExit(
            f"missing {split_manifest_path}. Cannot isolate development cells from "
            "holdout. Run the split step (scripts/00_build_canonical.py) first."
        )
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    split_manifest = pd.read_parquet(split_manifest_path)[["cell", "outer_role", "cv_fold"]]
    frame = frame.merge(split_manifest, on="cell", how="left", validate="many_to_one")
    frame = frame[frame["outer_role"].eq("development")].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("[bootstrap] No development cells found in split manifest.")
    return frame, json.loads(manifest.read_text())["feature_columns"]


def _pinn_results_dir(cfg: dict) -> Path:
    return (
        Path(cfg["paths"]["artifacts"]) / "results"
        / str((cfg.get("pinn") or {}).get("results_subdir", "pinn_unified_study"))
    )


def _collect_classical_oof(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Run CV on development folds; return per-anchor OOF errors for each model."""
    models = _classical_models(feature_columns)
    records = []
    has_anchor_id = "anchor_id" in frame.columns

    cv_folds = sorted(frame["cv_fold"].dropna().unique())
    for fold in cv_folds:
        fold = int(fold)
        train = frame[frame["cv_fold"].ne(fold) & frame["cv_fold"].notna()]
        test = frame[frame["cv_fold"].eq(fold)].reset_index(drop=True)
        for name, template in models.items():
            model = copy.deepcopy(template)
            model.fit(train[feature_columns], train[TARGET])
            preds = model.predict(test[feature_columns])
            for i in range(len(test)):
                rec: dict = {
                    "model": name,
                    "fold": fold,
                    "cell": str(test["cell"].iloc[i]),
                    "y_true": float(test[TARGET].iloc[i]),
                    "y_pred": float(preds[i]),
                    "error": float(preds[i] - test[TARGET].iloc[i]),
                }
                if has_anchor_id:
                    rec["anchor_id"] = str(test["anchor_id"].iloc[i])
                records.append(rec)

    return pd.DataFrame(records)


def _collect_pinn_oof(
    pinn_results_dir: Path,
    seed: int,
) -> pd.DataFrame | None:
    """Load per-anchor outer-val predictions for one PINN seed.

    predictions.parquet schema (written by trainer._prediction_frame):
        cell_id, anchor_id, fold, evaluation_role,
        true_delta_Q_Ah, predicted_delta_Q_Ah, ...
    """
    try:
        df = load_fold_predictions(pinn_results_dir, seed=seed)
    except FileNotFoundError as exc:
        print(f"[bootstrap] PINN seed {seed}: no predictions ({exc})")
        return None

    # Normalise column names to the shared schema.
    if "cell_id" in df.columns:
        df = df.rename(columns={"cell_id": "cell"})
    elif "cell" not in df.columns:
        print(f"[bootstrap] PINN seed {seed}: no cell/cell_id column; skipping")
        return None

    if "predicted_delta_Q_Ah" in df.columns:
        df["y_pred"] = df["predicted_delta_Q_Ah"].to_numpy(dtype=float)
    elif "y_pred" not in df.columns:
        print(f"[bootstrap] PINN seed {seed}: cannot derive y_pred; skipping")
        return None

    if "true_delta_Q_Ah" in df.columns:
        df["y_true"] = df["true_delta_Q_Ah"].to_numpy(dtype=float)
    elif "y_true" not in df.columns:
        print(f"[bootstrap] PINN seed {seed}: no y_true column; skipping")
        return None

    df["error"] = df["y_pred"] - df["y_true"]
    df["model"] = "NaPINN-Q"
    df["cell"] = df["cell"].astype(str)

    keep = ["model", "fold", "cell", "y_true", "y_pred", "error"]
    if "anchor_id" in df.columns:
        keep.append("anchor_id")
    return df[keep]


def _run_bootstrap_for_seed(
    classical_oof: pd.DataFrame,
    pinn_oof: pd.DataFrame | None,
    reference: str,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> pd.DataFrame:
    """Compute paired bootstrap between reference and all comparators."""
    frames = [classical_oof]
    if pinn_oof is not None:
        frames.append(pinn_oof)
    all_oof = pd.concat(frames, ignore_index=True)

    available_models = list(all_oof["model"].unique())
    if reference not in available_models:
        print(f"[bootstrap] reference '{reference}' not in OOF; available: {available_models}")
        return pd.DataFrame()

    ref_df = all_oof[all_oof["model"].eq(reference)].copy()
    comparators = [m for m in available_models if m != reference]

    rows = []
    for comparator in comparators:
        comp_df = all_oof[all_oof["model"].eq(comparator)].copy()

        # Align on fold + anchor_id when available; fall back to fold + cell.
        join_key = "anchor_id" if (
            "anchor_id" in ref_df.columns and "anchor_id" in comp_df.columns
        ) else "cell"
        merged = ref_df[["fold", join_key, "cell", "error"]].merge(
            comp_df[["fold", join_key, "error"]].rename(columns={"error": "error_comp"}),
            on=["fold", join_key], how="inner",
        )
        if len(merged) == 0:
            print(f"[bootstrap] no paired anchors for {reference} vs {comparator}; skipping")
            continue

        result = paired_cell_bootstrap(
            errors_a=merged["error"].to_numpy(dtype=float),
            errors_b=merged["error_comp"].to_numpy(dtype=float),
            cells=merged["cell"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci_level=ci_level,
        )

        if result["ci_high"] < 0:
            interp = f"{reference} significantly better (CI entirely < 0)"
        elif result["ci_low"] > 0:
            interp = f"{comparator} significantly better (CI entirely > 0)"
        else:
            interp = "statistically unresolved (CI crosses 0)"

        rows.append({
            "reference": reference,
            "comparator": comparator,
            "mean_paired_MAE_diff_Ah": result["mean_diff"],
            "ci_low_Ah": result["ci_low"],
            "ci_high_Ah": result["ci_high"],
            "p_reference_better": result["p_a_better"],
            "n_cells": result["n_cells"],
            "n_anchors": result["n_anchors"],
            "interpretation": interp,
        })

    return pd.DataFrame(rows)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--reference", type=str, default="NaPINN-Q",
                        help="model whose errors are the 'A' side; negative mean_diff "
                             "means the reference has lower MAE")
    parser.add_argument("--pinn-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--pinn-primary-seed", type=int, default=42)
    parser.add_argument("--pinn-architecture", type=str, default="NaPINN-Q")
    parser.add_argument("--pinn-preprocessing", type=str, default="unified")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load(cfg)
    pinn_dir = _pinn_results_dir(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "bootstrap"
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"[bootstrap] collecting classical OOF (cv_folds from split manifest) ...")
    classical_oof = _collect_classical_oof(frame, feature_columns)
    print(f"[bootstrap] {len(classical_oof)} classical anchor-predictions "
          f"({classical_oof['model'].nunique()} models)")

    # Primary seed (seed 42 is the headline comparison)
    pinn_primary = _collect_pinn_oof(pinn_dir, seed=args.pinn_primary_seed)
    if pinn_primary is not None:
        print(f"[bootstrap] PINN seed {args.pinn_primary_seed}: "
              f"{len(pinn_primary)} anchor-predictions")

    print(f"\n[bootstrap] === Primary comparison (seed {args.pinn_primary_seed}) ===")
    primary_table = _run_bootstrap_for_seed(
        classical_oof=classical_oof,
        pinn_oof=pinn_primary,
        reference=args.reference,
        n_bootstrap=args.n_bootstrap,
        seed=args.pinn_primary_seed,
        ci_level=args.ci_level,
    )
    if not primary_table.empty:
        primary_table.to_csv(out / "paired_bootstrap_seed42.csv", index=False)
        print(primary_table.to_string(index=False))
    else:
        print("[bootstrap] no primary comparison rows produced")

    # Sensitivity: all seeds
    sensitivity_rows = []
    for seed in args.pinn_seeds:
        pinn_seed_oof = _collect_pinn_oof(pinn_dir, seed=seed)
        seed_table = _run_bootstrap_for_seed(
            classical_oof=classical_oof,
            pinn_oof=pinn_seed_oof,
            reference=args.reference,
            n_bootstrap=args.n_bootstrap,
            seed=seed,
            ci_level=args.ci_level,
        )
        if not seed_table.empty:
            seed_table.insert(0, "pinn_seed", seed)
            sensitivity_rows.append(seed_table)

    if sensitivity_rows:
        sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
        sensitivity.to_csv(out / "paired_bootstrap_sensitivity.csv", index=False)
        print(f"\n[bootstrap] sensitivity (seeds {args.pinn_seeds}):")
        print(sensitivity[["pinn_seed", "comparator",
                            "mean_paired_MAE_diff_Ah", "ci_low_Ah", "ci_high_Ah",
                            "p_reference_better", "interpretation"]].to_string(index=False))

    print(f"\n[bootstrap] wrote {out}")
    return 0


def main() -> int:
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "bootstrap" / "logs"
    with run_log("17_run_paired_bootstrap", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
