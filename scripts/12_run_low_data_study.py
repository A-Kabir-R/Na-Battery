#!/usr/bin/env python3
"""Learning curves in units of training *cells*.

The data-efficiency claim the journal cares about is "physics constraints reduce
the aging-test burden", and the currency of that burden is cells, not anchors: a
cell is weeks of cycler time, an anchor is free once the cell exists.

For each outer fold the training pool is subsampled to 20/40/60/80/100% of its
cells, condition-stratified, repeated so the curve carries a band rather than a
single lucky draw. Test folds are held fixed across every budget so the curve
measures the budget and nothing else.

Writes, under ``artifacts/results/low_data/``:

    low_data_records.parquet   one row per (model, fold, fraction, repeat)
    low_data_curve.csv         median + IQR band per budget
    low_data_cells_to_reach.csv  cells each model needs to hit the reference error

Usage::

    python3 scripts/12_run_low_data_study.py
    python3 scripts/12_run_low_data_study.py --repeats 5 --models ConstantRate Ridge
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

import tempfile  # noqa: E402

from src.evaluation.low_data import (  # noqa: E402
    DEFAULT_FRACTIONS, LowDataStudy, cells_to_reach, summarize_curve,
)
from src.io.loaders import load_config  # noqa: E402
from src.pinn.inference import PINNInferenceSession, find_run_dir  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402
from src.models.degradation_baselines import build_degradation_baselines  # noqa: E402
from src.models.regressors import STUDY_RIDGE_ALPHA  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _statistical_models(feature_columns: list[str]):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
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
        # Frozen penalty, not RidgeCV: see STUDY_RIDGE_ALPHA. It matters more
        # here than anywhere else -- at the 20% budget a row-level LOO search
        # would tune alpha on a handful of cells' own anchors.
        "Ridge": wrap(Ridge(alpha=STUDY_RIDGE_ALPHA)),
        "ExtraTrees": wrap(ExtraTreesRegressor(n_estimators=400, random_state=42)),
        "GradientBoosting": wrap(GradientBoostingRegressor(random_state=42)),
        # The neural baseline needs a standardised target: the raw delta has a
        # scale of ~0.007 Ah and the optimiser stalls on it otherwise.
        "MLP": wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                 max_iter=8000, random_state=42), scale_target=True),
    }


def _pinn_results_dir(cfg: dict) -> Path:
    return (
        Path(cfg["paths"]["artifacts"]) / "results"
        / str((cfg.get("pinn") or {}).get("results_subdir", "pinn_unified_study"))
    )


def _make_pinn_fit_predict(
    pinn_results_dir: Path,
    cfg: dict,
    fold: int,
    pinn_seed: int,
    pinn_architecture: str,
    pinn_preprocessing: str,
    device: str,
    max_epochs: int | None,
):
    """Return a fit_predict callable for NaPINN-Q for use in LowDataStudy.

    The PINN is retrained from scratch on each training subset (proper_train
    cells only) so the learning curve reflects genuine data budget -- not a
    model that was already exposed to the full training set.

    The epoch count is read from the nested-selected best epoch stored in the
    seed-42 fold's status.json (if available), or falls back to max_epochs.
    Training uses the frozen loss configuration from config.yaml so no
    hyperparameter search occurs within each cell-budget draw.
    """
    import json as _json
    import time as _time
    from src.pinn.config_builder import build_trainer_config
    from src.pinn.dataset import apply_split_manifest, build_anchor_dataset
    from src.pinn.losses import LossWeights
    from src.pinn.trainer import FoldPaths, FoldTrainingError, train_fold
    from src.pinn.utils import resolve_device

    pinn_cfg = cfg["pinn"]
    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    real_device = resolve_device(device)

    # Read the nested-selected epoch from status.json; verify provenance.
    best_epoch: int | None = None
    seed42_status = (
        pinn_results_dir / "runs" / pinn_architecture / pinn_preprocessing
        / f"fold_{fold}" / f"seed_{pinn_seed}" / "status.json"
    )
    if seed42_status.exists():
        try:
            status = _json.loads(seed42_status.read_text())
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

    # Load the anchor dataset once; reuse across cell budgets for this fold.
    frame_raw = pd.read_parquet(features_dir / f"{pinn_preprocessing}.parquet")
    TARGET_COL = "delta_next_rpt_Q_Ah"
    dataset = build_anchor_dataset(
        frame_raw, preprocessing=pinn_preprocessing,
        stress_cfg=pinn_cfg["stress"],
        audit_cfg=pinn_cfg.get("audit"),
        audit_path=None,
        features_dir=features_dir,
    )
    loss_weights = LossWeights.from_mapping(pinn_cfg["losses"])

    def fit_predict(train_sub: pd.DataFrame, test_sub: pd.DataFrame) -> np.ndarray:
        """Train NaPINN-Q on train_sub cells, predict TARGET on test_sub."""
        train_cells = set(train_sub["cell"].astype(str).unique())
        test_cells = set(test_sub["cell"].astype(str).unique())
        all_cells = train_cells | test_cells

        # Build a fake split manifest: train cells → cv_fold=1, test → cv_fold=0.
        # fold_indices(frame, fold=0) → outer_train = cv_fold != 0,
        #                                outer_val   = cv_fold == 0.
        anchor_frame = dataset.frame.copy()
        anchor_frame = anchor_frame[
            anchor_frame["cell"].astype(str).isin(all_cells)
        ].reset_index(drop=True)
        anchor_frame["outer_role"] = "development"
        anchor_frame["cv_fold"] = anchor_frame["cell"].astype(str).map(
            lambda c: 0 if c in test_cells else 1
        )

        with tempfile.TemporaryDirectory(prefix="pinn_ld_") as tmp:
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
                result = train_fold(
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
                print(f"[low-data/pinn] training failed: {exc}")
                return np.full(len(test_sub), float("nan"))

            # Load the trained checkpoint and predict on test_sub.
            try:
                from src.pinn.inference import prepare_anchor_frame
                session = PINNInferenceSession(run_dir, device=real_device)
                test_rows = dataset.frame[
                    dataset.frame["cell"].astype(str).isin(test_cells)
                ]
                if test_rows.empty:
                    return np.full(len(test_sub), float("nan"))
                return session.predict_delta_q_ah(test_rows)
            except Exception as exc:
                print(f"[low-data/pinn] inference failed: {exc}")
                return np.full(len(test_sub), float("nan"))

    return fit_predict


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
    frame = frame[frame["outer_role"].eq("development")].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("[low-data] No development cells found in split manifest.")
    return frame, json.loads(manifest.read_text())["feature_columns"]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20,
                        help="cell draws per budget (default 20)")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of model names; default is all")
    parser.add_argument("--reference-model", default="ExtraTrees",
                        help="model whose full-data error defines the threshold")
    parser.add_argument("--pinn", action="store_true",
                        help="include NaPINN-Q in the learning curve. "
                             "Trains a fresh PINN per (fold, fraction, repeat) — "
                             "GPU recommended; use --pinn-max-epochs to limit cost.")
    parser.add_argument("--pinn-seed", type=int, default=42)
    parser.add_argument("--pinn-architecture", type=str, default="NaPINN-Q")
    parser.add_argument("--pinn-preprocessing", type=str, default="unified")
    parser.add_argument("--pinn-max-epochs", type=int, default=None,
                        help="override epoch count for low-data PINN training; "
                             "defaults to the nested-selected best epoch from the "
                             "seed-42 status.json (or config max_epochs as fallback)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "low_data"
    )
    out.mkdir(parents=True, exist_ok=True)

    models: dict[str, object] = {
        **build_degradation_baselines(),
        **_statistical_models(feature_columns),
    }
    if args.models:
        unknown = set(args.models) - (set(models) | ({"NaPINN-Q"} if args.pinn else set()))
        if unknown:
            raise SystemExit(f"unknown model(s): {sorted(unknown)}")
        models = {name: models[name] for name in args.models if name != "NaPINN-Q"}

    print(f"[low-data] anchors={len(frame)} cells={frame['cell'].nunique()} "
          f"models={len(models) + (1 if args.pinn else 0)} repeats={args.repeats}")
    if args.pinn:
        print("[low-data] NaPINN-Q: will retrain from scratch at each cell budget "
              "(slow; one GPU run per fold × fraction × repeat)")

    records: list[pd.DataFrame] = []
    cv_folds = sorted(frame["cv_fold"].dropna().unique())
    for fold in cv_folds:
        test = frame[frame["cv_fold"].eq(fold)].reset_index(drop=True)
        pool = frame[frame["cv_fold"].ne(fold) & frame["cv_fold"].notna()].reset_index(drop=True)
        print(f"[low-data] fold {int(fold)}: pool {pool['cell'].nunique()} cells, "
              f"test {test['cell'].nunique()} cells")

        fold_int = int(fold)
        for name, template in models.items():
            def fit_predict(train, test_frame, _template=template, _name=name):
                model = copy.deepcopy(_template)
                if _name in build_degradation_baselines():
                    model.fit(train, train[TARGET].to_numpy(dtype=float))
                    return model.predict(test_frame)
                model.fit(train[feature_columns], train[TARGET])
                return model.predict(test_frame[feature_columns])

            study = LowDataStudy(
                fractions=tuple(args.fractions), repeats=args.repeats, seed=fold_int,
            )
            records.append(study.run(
                pool, test, TARGET, fit_predict, model_name=name, fold=fold_int,
            ))

        # ---- NaPINN-Q learning curve ----------------------------------------
        if args.pinn:
            pinn_dir = _pinn_results_dir(cfg)
            pinn_fit_predict = _make_pinn_fit_predict(
                pinn_dir, cfg, fold_int,
                pinn_seed=args.pinn_seed,
                pinn_architecture=args.pinn_architecture,
                pinn_preprocessing=args.pinn_preprocessing,
                device=args.device,
                max_epochs=args.pinn_max_epochs,
            )

            def pinn_fit_predict_wrapped(train, test_frame):
                return pinn_fit_predict(train, test_frame)

            pinn_study = LowDataStudy(
                fractions=tuple(args.fractions),
                repeats=args.repeats,
                seed=fold_int,
            )
            records.append(pinn_study.run(
                pool, test, TARGET,
                pinn_fit_predict_wrapped,
                model_name="NaPINN-Q",
                fold=fold_int,
            ))

    if not records:
        print("[low-data] nothing ran")
        return 1

    all_records = pd.concat(records, ignore_index=True)
    all_records.to_parquet(out / "low_data_records.parquet", index=False)
    failures = int((all_records["status"] != "ok").sum())
    if failures:
        print(f"[low-data] WARNING: {failures} failed fits (see records)")

    curve = summarize_curve(all_records)
    curve.to_csv(out / "low_data_curve.csv", index=False)

    # How many cells each model needs to match the reference model's full-data
    # error. This is the sentence the paper reduces to: "the proposed model
    # reaches the same accuracy with N fewer cells."
    reference = curve[
        curve["model"].eq(args.reference_model) & (curve["fraction"] >= 1.0)
    ]
    rows = []
    if reference.empty:
        print(f"[low-data] reference model {args.reference_model} not in curve")
    else:
        threshold = float(reference["median"].iloc[0])
        for name in curve["model"].unique():
            rows.append({
                "model": name,
                "threshold_cell_macro_MAE": threshold,
                "reference_model": args.reference_model,
                "cells_to_reach": cells_to_reach(
                    curve[curve["model"].eq(name)], threshold,
                ),
            })
        pd.DataFrame(rows).to_csv(out / "low_data_cells_to_reach.csv", index=False)
        print(f"\n[low-data] threshold = {threshold:.5f} Ah "
              f"({args.reference_model} at full data)")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + curve.to_string(index=False))
    print(f"\n[low-data] wrote {out}")
    return 0


def main() -> int:
    """Run under a capture log so a finished run is still debuggable.

    Everything these studies printed used to go to stdout and nowhere else.
    """
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "low_data" / "logs"
    with run_log("12_run_low_data_study", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
