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

For NaPINN-Q, the conformal protocol mirrors the classical arm exactly:
  1. Split the outer-training pool into proper-train and calibration cells.
  2. Retrain NaPINN-Q from scratch on proper-train cells only (frozen loss config,
     nested-selected epoch count from the full seed-42 run).
  3. Predict on calibration cells → nonconformity scores.
  4. Apply split conformal to outer-val cells.
This is the only valid protocol: the final-refit model trains on all outer-training
cells (including inner-calibration cells), so its residuals on any outer-training
cell are not exchangeable with outer-val residuals.

Writes, under ``artifacts/results/uncertainty/``:

    uncertainty_predictions.parquet  per-row predictions with lower/upper bounds
    uncertainty_metrics.csv          PICP / MPIW / interval score, pooled
    coverage_by_condition.csv        the number that actually matters
    coverage_by_cell.csv

Usage::

    python3 scripts/13_run_uncertainty_study.py --alpha 0.1 --calibration-fraction 0.45
    python3 scripts/13_run_uncertainty_study.py --pinn --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile  # noqa: E402
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
from src.pinn.inference import PINNInferenceSession, prepare_anchor_frame  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402

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
    split_manifest_path = artifacts / "splits" / "split_manifest.parquet"
    if not unified.exists():
        raise SystemExit(
            f"missing {unified}. Run scripts/01_build_features.py first, or point "
            f"SIB_ARTIFACTS at a built artifacts directory."
        )
    if not split_manifest_path.exists():
        raise SystemExit(
            f"missing {split_manifest_path}. Cannot isolate development cells from "
            "holdout. Run the split step (scripts/00_build_canonical.py) first."
        )
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    split_manifest = pd.read_parquet(split_manifest_path)[["cell", "outer_role", "cv_fold"]]
    frame = frame.merge(split_manifest, on="cell", how="left", validate="many_to_one")
    holdout_count = int(frame["outer_role"].eq("holdout").sum())
    if holdout_count > 0:
        raise SystemExit(
            f"[uq] {holdout_count} holdout rows found after merge — "
            "aborting to prevent data leakage."
        )
    frame = frame[frame["outer_role"].eq("development")].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("[uq] No development cells found in split manifest.")
    return frame, json.loads(manifest.read_text())["feature_columns"]


def _pinn_results_dir(cfg: dict) -> Path:
    return (
        Path(cfg["paths"]["artifacts"]) / "results"
        / str((cfg.get("pinn") or {}).get("results_subdir", "pinn_unified_study"))
    )


def _pinn_train_and_predict(
    cfg: dict,
    fold: int,
    proper_train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    pinn_seed: int,
    pinn_architecture: str,
    pinn_preprocessing: str,
    device: str,
    max_epochs: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Retrain PINN on proper_train cells; predict calibration and test cells.

    Protocol mirrors the classical arm exactly:
      - proper_train cells → PINN training (scaler fitted here only)
      - calibration cells  → nonconformity scores (model never saw these)
      - test cells         → conformal interval evaluation

    Returns (cal_y_pred, test_y_pred) as delta predictions (u_next - u_current),
    or None if training or checkpoint loading fails.
    """
    import json as _json
    from src.pinn.config_builder import build_trainer_config
    from src.pinn.dataset import build_anchor_dataset
    from src.pinn.losses import LossWeights
    from src.pinn.trainer import FoldPaths, FoldTrainingError, train_fold
    from src.pinn.utils import resolve_device

    pinn_cfg = cfg["pinn"]
    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    real_device = resolve_device(device)
    pinn_results_dir = _pinn_results_dir(cfg)

    # Frozen epoch count from nested-selection status.json; verify provenance.
    best_epoch: int | None = None
    seed_status = (
        pinn_results_dir / "runs" / pinn_architecture / pinn_preprocessing
        / f"fold_{fold}" / f"seed_{pinn_seed}" / "status.json"
    )
    if seed_status.exists():
        try:
            status = _json.loads(seed_status.read_text())
            if (status.get("refit_epoch_selection_source") == "nested_inner_folds"
                    and status.get("two_phase_refit_used")
                    and status.get("status") == "completed"):
                best_epoch = int(status["refit_selected_epoch"])
            else:
                best_epoch = int(status.get("refit_selected_epoch",
                                            status.get("best_epoch", 0)))
        except Exception:
            best_epoch = None
    if best_epoch is None or best_epoch <= 0:
        best_epoch = max_epochs or int(pinn_cfg["training"]["maximum_epochs"])

    frame_raw = pd.read_parquet(features_dir / f"{pinn_preprocessing}.parquet")
    dataset = build_anchor_dataset(
        frame_raw, preprocessing=pinn_preprocessing,
        stress_cfg=pinn_cfg["stress"],
        audit_cfg=pinn_cfg.get("audit"),
        audit_path=None,
        features_dir=features_dir,
    )
    loss_weights = LossWeights.from_mapping(pinn_cfg["losses"])

    train_cells = set(proper_train["cell"].astype(str).unique())
    cal_cells = set(calibration["cell"].astype(str).unique())
    test_cells = set(test["cell"].astype(str).unique())
    all_cells = train_cells | cal_cells | test_cells

    # cv_fold=1 → proper_train (outer_train); cv_fold=0 → cal+test (outer_val)
    anchor_frame = dataset.frame.copy()
    anchor_frame = anchor_frame[
        anchor_frame["cell"].astype(str).isin(all_cells)
    ].reset_index(drop=True)
    anchor_frame["outer_role"] = "development"
    anchor_frame["cv_fold"] = anchor_frame["cell"].astype(str).map(
        lambda c: 1 if c in train_cells else 0
    )

    with tempfile.TemporaryDirectory(prefix="pinn_uq_") as tmp:
        run_dir = Path(tmp)
        paths = FoldPaths(root=run_dir)
        trainer_cfg = build_trainer_config(
            pinn_cfg, architecture=pinn_architecture,
            preprocessing=pinn_preprocessing,
            fold=0, seed=pinn_seed, device=real_device,
            max_epochs=best_epoch,
            two_phase_refit=True,
            forced_best_epoch=best_epoch,
        )
        try:
            train_fold(
                dataset=dataset, frame=anchor_frame,
                config=trainer_cfg, fold_paths=paths,
                loss_weights=loss_weights,
                u_max_source=str(pinn_cfg["bounds"]["u_max_source"]),
                u_max_constant=float(pinn_cfg["bounds"]["u_max_constant"]),
                u_max_margin=float(pinn_cfg["bounds"]["u_max_margin"]),
                tolerance_source=str(pinn_cfg["monotonicity"]["tolerance_source"]),
                tolerance_constant=float(pinn_cfg["monotonicity"]["tolerance_constant"]),
                tolerance_quantile=float(pinn_cfg["monotonicity"]["tolerance_quantile"]),
                force=True,
            )
        except FoldTrainingError as exc:
            print(f"[uq/pinn] fold {fold}: training failed: {exc}")
            return None

        try:
            session = PINNInferenceSession(run_dir, device=real_device)
        except FileNotFoundError as exc:
            print(f"[uq/pinn] fold {fold}: checkpoint not found: {exc}")
            return None

        cal_anchor = prepare_anchor_frame(calibration, dataset)
        test_anchor = prepare_anchor_frame(test, dataset)
        cal_y_pred = session.predict_delta_q_ah(cal_anchor)
        test_y_pred = session.predict_delta_q_ah(test_anchor)

    return cal_y_pred, test_y_pred


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
    parser.add_argument("--pinn", action="store_true",
                        help="add NaPINN-Q by retraining per fold on proper-train cells "
                             "(GPU recommended; mirrors the classical conformal arm exactly)")
    parser.add_argument("--pinn-seed", type=int, default=42)
    parser.add_argument("--pinn-architecture", type=str, default="NaPINN-Q")
    parser.add_argument("--pinn-preprocessing", type=str, default="unified")
    parser.add_argument("--pinn-max-epochs", type=int, default=None,
                        help="override epoch count; default reads nested-selected epoch "
                             "from seed-42 status.json")
    parser.add_argument("--device", type=str, default="cpu")
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

    cv_folds = sorted(frame["cv_fold"].dropna().unique())
    for fold in cv_folds:
        fold = int(fold)
        test = frame[frame["cv_fold"].eq(fold)].reset_index(drop=True)
        pool = frame[frame["cv_fold"].ne(fold) & frame["cv_fold"].notna()].reset_index(drop=True)
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

        # ---- NaPINN-Q: same proper-train/calibration/test split as classical ---
        if args.pinn:
            print(f"[uq/pinn] fold {fold}: retraining on "
                  f"{proper['cell'].nunique()} proper-train cells, "
                  f"calibrating on {n_calibration_cells} cells")
            pinn_result = _pinn_train_and_predict(
                cfg=cfg, fold=fold,
                proper_train=proper, calibration=calibration, test=test,
                pinn_seed=args.pinn_seed,
                pinn_architecture=args.pinn_architecture,
                pinn_preprocessing=args.pinn_preprocessing,
                device=args.device,
                max_epochs=args.pinn_max_epochs,
            )
            if pinn_result is not None:
                cal_y_pred_pinn, test_y_pred_pinn = pinn_result

                cal_frame_pinn = calibration[["cell", "condition"]].copy()
                cal_frame_pinn["y_true"] = calibration[TARGET].to_numpy(dtype=float)
                cal_frame_pinn["y_pred"] = cal_y_pred_pinn
                try:
                    conformal_pinn = GroupedConformal(
                        alpha=args.alpha, normalize=args.normalize,
                        score_reduction=args.score_reduction,
                    ).fit(cal_frame_pinn)
                except ConformalError as exc:
                    print(f"[uq/pinn] fold {fold}: conformal failed ({exc}); skipping")
                    conformal_pinn = None

                if conformal_pinn is not None:
                    test_frame_pinn = test[["cell", "condition"]].copy()
                    test_frame_pinn["y_true"] = test[TARGET].to_numpy(dtype=float)
                    test_frame_pinn["y_pred"] = test_y_pred_pinn
                    test_frame_pinn["ensemble_std"] = float("nan")
                    test_frame_pinn["n_members"] = 1
                    test_frame_pinn["model"] = "NaPINN-Q"
                    test_frame_pinn["fold"] = fold
                    test_frame_pinn["alpha"] = args.alpha
                    test_frame_pinn["n_calibration_cells"] = n_calibration_cells
                    if "anchor_id" in test.columns:
                        test_frame_pinn["anchor_id"] = test["anchor_id"].to_numpy()
                    intervals_pinn = conformal_pinn.predict_interval(test_frame_pinn)
                    predictions.append(pd.concat([test_frame_pinn, intervals_pinn], axis=1))

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
    if args.pinn:
        print("\n[uq/pinn] protocol: retrain on proper-train cells, calibrate on "
              "held-back calibration cells, evaluate on outer-val cells "
              "(mirrors classical conformal arm; no exchangeability violation)")
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
