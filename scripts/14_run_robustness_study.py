#!/usr/bin/env python3
"""Input-perturbation robustness, with interval widening as the pass criterion.

A model that degrades gracefully is not the same as one that degrades honestly.
The requirement here is that prediction intervals **widen** as inputs degrade: a
model whose accuracy falls while its confidence holds is worse than useless in
deployment, because nothing signals that its output should be distrusted.

Perturbations act on the built feature table, not on raw ``.ird`` signals. That
is a deliberate scope limit and the manuscript must state it: a current-gain
error applied here propagates only through features that already encode current,
not through re-integration of the waveform.

The model is fitted once on clean data and then *met* with corrupted inputs. It
is not retrained on them -- retraining would measure a different, easier question.

Writes, under ``artifacts/results/robustness/``:

    robustness_metrics.csv   MAE ratio and interval-width ratio per perturbation
    robustness_summary.csv   per model, how many perturbations widened correctly

Usage::

    python3 scripts/14_run_robustness_study.py
    python3 scripts/14_run_robustness_study.py --temperature-offset 5 --no-intervals
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

from src.evaluation.conformal import (  # noqa: E402
    ConformalError, GroupedConformal, split_calibration_cells,
)
from src.evaluation.robustness import build_perturbations, run_robustness  # noqa: E402
from src.io.loaders import load_config  # noqa: E402
from src.models.regressors import STUDY_RIDGE_ALPHA  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402
from src.splits.group_kfold import make_folds  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _models(feature_columns: list[str]):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

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
        # Frozen penalty, not RidgeCV: see STUDY_RIDGE_ALPHA.
        "Ridge": wrap(Ridge(alpha=STUDY_RIDGE_ALPHA)),
        "ExtraTrees": wrap(ExtraTreesRegressor(n_estimators=400, random_state=42)),
        "MLP": wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                 max_iter=8000, random_state=42), scale_target=True),
    }


def _load(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    artifacts = Path(cfg["paths"]["artifacts"])
    unified = artifacts / "features" / "unified.parquet"
    manifest = artifacts / "features" / "unified_feature_manifest.json"
    if not unified.exists():
        raise SystemExit(
            f"missing {unified}. Run scripts/01_build_features.py first, or point "
            f"SIB_ARTIFACTS at a built artifacts directory."
        )
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    return frame, json.loads(manifest.read_text())["feature_columns"]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="interval level for the widening check")
    parser.add_argument("--calibration-fraction", type=float, default=0.35)
    parser.add_argument("--temperature-offset", type=float, default=2.0)
    parser.add_argument("--voltage-noise", type=float, default=0.005)
    parser.add_argument("--current-gain", type=float, default=1.02)
    parser.add_argument("--no-intervals", action="store_true",
                        help="skip conformal; report error degradation only")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "robustness"
    )
    out.mkdir(parents=True, exist_ok=True)

    perturbations = build_perturbations(
        temperature_offset_C=args.temperature_offset,
        voltage_noise_V=args.voltage_noise,
        current_gain=args.current_gain,
    )
    print(f"[robust] {len(perturbations)} perturbations x {len(_models(feature_columns))} "
          f"models x {args.folds} folds")

    reports: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(
        make_folds(frame, n_splits=args.folds)
    ):
        pool = frame.iloc[train_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)

        for name, template in _models(feature_columns).items():
            model = copy.deepcopy(template)
            conformal = None
            if not args.no_intervals:
                try:
                    proper_idx, calibration_idx = split_calibration_cells(
                        pool, calibration_fraction=args.calibration_fraction,
                        seed=fold,
                    )
                    proper = pool.iloc[proper_idx]
                    calibration = pool.iloc[calibration_idx]
                    model.fit(proper[feature_columns], proper[TARGET])
                    calibration_frame = calibration[["cell", "condition"]].copy()
                    calibration_frame["y_true"] = calibration[TARGET].to_numpy(float)
                    calibration_frame["y_pred"] = model.predict(
                        calibration[feature_columns]
                    )
                    conformal = GroupedConformal(
                        alpha=args.alpha, normalize=True,
                    ).fit(calibration_frame)
                except ConformalError as error:
                    print(f"[robust] fold {fold} {name}: no intervals ({error})")
                    conformal = None
                    model = copy.deepcopy(template)
                    model.fit(pool[feature_columns], pool[TARGET])
            else:
                model.fit(pool[feature_columns], pool[TARGET])

            def predict(subject: pd.DataFrame) -> np.ndarray:
                return model.predict(subject[feature_columns])

            half_width = None
            if conformal is not None:
                def half_width(subject: pd.DataFrame) -> np.ndarray:
                    scored = subject[["cell"]].copy()
                    scored["y_pred"] = predict(subject)
                    return conformal.predict_interval(scored)["half_width"].to_numpy()

            report = run_robustness(
                test, TARGET, predict, perturbations=perturbations,
                interval_half_width=half_width, seed=fold,
            )
            report.insert(0, "model", name)
            report.insert(1, "fold", fold)
            reports.append(report)

    if not reports:
        print("[robust] nothing ran")
        return 1

    metrics = pd.concat(reports, ignore_index=True)
    metrics.to_csv(out / "robustness_metrics.csv", index=False)

    failures = int((metrics["status"] != "ok").sum())
    if failures:
        print(f"[robust] {failures} perturbation runs failed (recorded in output)")

    perturbed = metrics[metrics["perturbation"].ne("none") & metrics["status"].eq("ok")]
    summary = (
        perturbed.groupby(["model", "perturbation"])
        .agg(mean_MAE_ratio=("MAE_ratio", "mean"),
             mean_width_ratio=("width_ratio", "mean"),
             folds=("fold", "nunique"))
        .reset_index()
        .sort_values(["model", "mean_MAE_ratio"], ascending=[True, False])
    )
    summary.to_csv(out / "robustness_summary.csv", index=False)

    print("\n[robust] worst degradations (MAE ratio > 1 means the perturbation hurt)")
    print(summary.head(15).to_string(index=False))

    if not args.no_intervals and perturbed["widened"].notna().any():
        widened = (
            perturbed.groupby("model")["widened"]
            .agg(["sum", "count"]).reset_index()
            .rename(columns={"sum": "widened", "count": "perturbations"})
        )
        print("\n[robust] did the intervals admit the loss of information?")
        print(widened.to_string(index=False))
        print("  (a model that loses a diagnostic and stays equally confident is "
              "the failure mode this study exists to catch)")

    print(f"\n[robust] wrote {out}")
    return 0


def main() -> int:
    """Run under a capture log so a finished run is still debuggable.

    Everything these studies printed used to go to stdout and nowhere else.
    """
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "robustness" / "logs"
    with run_log("14_run_robustness_study", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
