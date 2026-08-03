#!/usr/bin/env python3
"""Cell-grouped conformal intervals and coverage reporting.

Accuracy differences between the top model families are not statistically
resolvable on this dataset -- a paired cell-clustered bootstrap puts ExtraTrees
vs Ridge at [-0.00041, 0.00083] Ah, crossing zero. Calibrated uncertainty is
therefore where the methodological contribution lands.

Two facts this script is built around:

* The exchangeability unit is the **cell**. Calibrating on a random subset of
  anchors leaks a cell's residual scale into its own interval.
* Split conformal needs ``1/alpha - 1`` calibration *cells*. With ~21 cells in an
  outer-training fold, 90% intervals need a calibration fraction of about 0.45,
  and 95% intervals are simply not certifiable here. The script checks this up
  front and says so rather than emitting a mislabelled interval.

Writes, under ``artifacts/results/uncertainty/``:

    uncertainty_predictions.parquet  per-row predictions with lower/upper bounds
    uncertainty_metrics.csv          PICP / MPIW / interval score, pooled
    coverage_by_condition.csv        the number that actually matters
    coverage_by_cell.csv

Usage::

    python3 scripts/13_run_uncertainty_study.py --alpha 0.1 --calibration-fraction 0.45
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
    ConformalError, GroupedConformal, coverage_by_group, interval_metrics,
    max_supported_level, min_calibration_units, split_calibration_cells,
)
from src.io.loaders import load_config  # noqa: E402
from src.models.regressors import STUDY_RIDGE_ALPHA  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402
from src.splits.group_kfold import make_folds  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _models(feature_columns: list[str], seeds: tuple[int, ...]):
    """Point models plus a deep ensemble over seeds.

    The ensemble exists because seeds are currently trained and reported
    separately, so the published figure is one arbitrarily-designated seed. An
    ensemble mean removes the temptation to report the seed that happened to win.
    """
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
        # Frozen penalty, not RidgeCV: see STUDY_RIDGE_ALPHA. Ridge is
        # deterministic, so its "ensemble" is correctly a single member.
        "Ridge": [wrap(Ridge(alpha=STUDY_RIDGE_ALPHA))],
        "ExtraTrees": [
            wrap(ExtraTreesRegressor(n_estimators=400, random_state=seed))
            for seed in seeds
        ],
        "MLP": [
            wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                              max_iter=8000, random_state=seed), scale_target=True)
            for seed in seeds
        ],
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
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="miscoverage rate; 0.1 gives 90%% intervals")
    parser.add_argument("--calibration-fraction", type=float, default=0.45)
    parser.add_argument("--score-reduction", default="max",
                        choices=["max", "mean", "median"],
                        help="how a calibration cell's anchors become one score")
    parser.add_argument("--normalize", action="store_true",
                        help="locally adaptive intervals scaled by predicted "
                             "magnitude (defined for unseen cells)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "uncertainty"
    )
    out.mkdir(parents=True, exist_ok=True)

    required = min_calibration_units(args.alpha)
    print(f"[uq] nominal {100 * (1 - args.alpha):.0f}% intervals need "
          f">= {required} calibration CELLS")

    models = _models(feature_columns, tuple(args.seeds))
    predictions: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(
        make_folds(frame, n_splits=args.folds)
    ):
        pool = frame.iloc[train_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)
        try:
            proper_idx, calibration_idx = split_calibration_cells(
                pool, calibration_fraction=args.calibration_fraction, seed=fold,
            )
        except ConformalError as error:
            print(f"[uq] fold {fold}: {error}")
            continue

        proper = pool.iloc[proper_idx]
        calibration = pool.iloc[calibration_idx]
        n_calibration_cells = calibration["cell"].nunique()
        if n_calibration_cells < required:
            print(
                f"[uq] fold {fold}: {n_calibration_cells} calibration cells "
                f"cannot certify {100 * (1 - args.alpha):.0f}% "
                f"(max {100 * max_supported_level(n_calibration_cells):.1f}%). "
                f"Raise --calibration-fraction or --alpha."
            )
            continue

        for name, members in models.items():
            fitted = []
            for member in members:
                model = copy.deepcopy(member)
                model.fit(proper[feature_columns], proper[TARGET])
                fitted.append(model)

            def ensemble_predict(subject: pd.DataFrame) -> np.ndarray:
                stacked = np.column_stack([
                    model.predict(subject[feature_columns]) for model in fitted
                ])
                return stacked.mean(axis=1), stacked.std(axis=1)

            calibration_frame = calibration[["cell", "condition"]].copy()
            calibration_mean, _ = ensemble_predict(calibration)
            calibration_frame["y_true"] = calibration[TARGET].to_numpy(dtype=float)
            calibration_frame["y_pred"] = calibration_mean

            conformal = GroupedConformal(
                alpha=args.alpha,
                normalize=args.normalize,
                score_reduction=args.score_reduction,
            ).fit(calibration_frame)

            test_frame = test[["cell", "condition"]].copy()
            test_mean, test_spread = ensemble_predict(test)
            test_frame["y_true"] = test[TARGET].to_numpy(dtype=float)
            test_frame["y_pred"] = test_mean
            test_frame["ensemble_std"] = test_spread
            test_frame["n_members"] = len(fitted)
            test_frame["model"] = name
            test_frame["fold"] = fold
            test_frame["alpha"] = args.alpha
            test_frame["n_calibration_cells"] = n_calibration_cells
            if "anchor_id" in test.columns:
                test_frame["anchor_id"] = test["anchor_id"].to_numpy()

            intervals = conformal.predict_interval(test_frame)
            predictions.append(pd.concat([test_frame, intervals], axis=1))

    if not predictions:
        print("[uq] no fold could be calibrated at this level")
        return 1

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(out / "uncertainty_predictions.parquet", index=False)

    pooled = []
    for name, group in all_predictions.groupby("model", sort=True):
        pooled.append({"model": name, **interval_metrics(group, alpha=args.alpha)})
    pooled_frame = pd.DataFrame(pooled)
    pooled_frame.to_csv(out / "uncertainty_metrics.csv", index=False)

    by_condition = coverage_by_group(
        all_predictions, ("model", "condition"), alpha=args.alpha,
    )
    by_condition.to_csv(out / "coverage_by_condition.csv", index=False)
    coverage_by_group(
        all_predictions, ("model", "cell"), alpha=args.alpha,
    ).to_csv(out / "coverage_by_cell.csv", index=False)

    print("\n[uq] pooled coverage")
    print(pooled_frame.to_string(index=False))
    print("\n[uq] worst-covered conditions (marginal coverage can look fine "
          "while a whole condition is uncovered)")
    print(
        by_condition.sort_values("PICP").head(10)[
            ["model", "condition", "n", "PICP", "MPIW"]
        ].to_string(index=False)
    )
    print(f"\n[uq] wrote {out}")
    return 0


def main() -> int:
    """Run under a capture log so a finished run is still debuggable.

    Everything these studies printed used to go to stdout and nowhere else.
    """
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "uncertainty" / "logs"
    with run_log("13_run_uncertainty_study", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
