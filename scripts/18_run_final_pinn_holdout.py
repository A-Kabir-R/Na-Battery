#!/usr/bin/env python3
"""Final exploratory locked-holdout evaluation for NaPINN-Q.

This script is run EXACTLY ONCE after the full development pipeline is frozen.
It must never be used to tune losses, architecture, epochs, or preprocessing.

Protocol
--------
1. Read the nested-selected epoch decisions from the completed fold runs.
   The epoch is the *consensus* over inner folds for seed 42, already written
   by script 15. Use the median over outer folds as the final epoch count.
2. Load all 26 development cells. Fit scaler on them only.
3. Train NaPINN-Q for the prespecified epoch count, no early stopping.
4. Predict the nine locked-holdout cells exactly once.
5. Compute cell-macro MAE and condition-macro MAE on development and holdout.
6. Run the paired cell-clustered bootstrap against matched classical predictions.
7. Write all outputs labeled ``exploratory_locked_holdout``.

Do NOT run this script before the development pipeline is complete and frozen.
Do NOT re-run this script after inspecting the holdout results.

Writes, under ``artifacts/results/pinn_holdout/``:

    pinn_holdout_predictions.parquet   per-anchor predictions (dev + holdout)
    pinn_holdout_metrics.csv           MAE per split (development / holdout)
    pinn_holdout_bootstrap.csv         paired bootstrap vs classical
    pinn_holdout_epoch_decision.json   which epoch was used and why

Usage::

    python3 scripts/18_run_final_pinn_holdout.py --dry-run
    python3 scripts/18_run_final_pinn_holdout.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap import paired_cell_bootstrap  # noqa: E402
from src.io.loaders import load_config  # noqa: E402
from src.pinn.inference import PINNInferenceSession, prepare_anchor_frame  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _pinn_results_dir(cfg: dict) -> Path:
    return (
        Path(cfg["paths"]["artifacts"]) / "results"
        / str((cfg.get("pinn") or {}).get("results_subdir", "pinn_unified_study"))
    )


def _read_nested_epoch(
    pinn_results_dir: Path,
    architecture: str,
    preprocessing: str,
    n_outer_folds: int,
    seed: int,
) -> tuple[int, dict]:
    """Read nested-selected epochs from each fold's status.json.

    Returns (median_epoch, details_dict).
    The median over outer folds is used as the final training epoch count to
    avoid over-fitting the epoch decision to any single fold's inner curves.
    """
    runs_root = pinn_results_dir / "runs" / architecture / preprocessing
    epochs = []
    details: dict = {}
    rejected: list[str] = []
    for fold in range(n_outer_folds):
        status_path = runs_root / f"fold_{fold}" / f"seed_{seed}" / "status.json"
        if not status_path.exists():
            print(f"[holdout] fold {fold}: status.json missing; skipping epoch")
            rejected.append(f"fold_{fold}: status.json missing")
            continue
        status = json.loads(status_path.read_text())
        if not (status.get("refit_epoch_selection_source") == "nested_inner_folds"
                and status.get("two_phase_refit_used")
                and status.get("status") == "completed"):
            msg = (f"fold_{fold}: selection_source="
                   f"{status.get('refit_epoch_selection_source')!r} "
                   f"two_phase_refit={status.get('two_phase_refit_used')} "
                   f"status={status.get('status')!r}")
            print(f"[holdout] {msg} — not a valid nested-selected run; skipping")
            rejected.append(msg)
            continue
        ep = int(status["refit_selected_epoch"])
        if ep > 0:
            epochs.append(ep)
            details[f"fold_{fold}"] = ep

    if not epochs:
        raise SystemExit(
            "[holdout] No valid nested-selected epochs found. "
            "Run scripts/15_run_nested_pinn_selection.py with two_phase_refit=True "
            "first. Rejected folds: " + "; ".join(rejected)
        )

    median_epoch = int(np.median(epochs))
    details["median_epoch_used"] = median_epoch
    details["per_fold_epochs"] = {k: v for k, v in details.items() if k.startswith("fold_")}
    details["rejected_folds"] = rejected
    print(f"[holdout] nested epochs per fold: {details['per_fold_epochs']}")
    print(f"[holdout] median epoch (used for final training): {median_epoch}")
    return median_epoch, details


def _load_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (development_frame, holdout_frame) from unified.parquet + split manifest."""
    artifacts = Path(cfg["paths"]["artifacts"])
    unified = artifacts / "features" / "unified.parquet"
    split_manifest_path = artifacts / "splits" / "split_manifest.parquet"

    if not unified.exists():
        raise SystemExit(f"missing {unified}. Run scripts/01_build_features.py first.")
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)

    if not split_manifest_path.exists():
        raise SystemExit(
            f"missing {split_manifest_path}. "
            "Cannot determine which cells are locked holdout."
        )
    split_manifest = pd.read_parquet(split_manifest_path)[["cell", "outer_role"]]
    frame = frame.merge(split_manifest, on="cell", how="left")

    dev = frame[frame["outer_role"].eq("development")].reset_index(drop=True)
    holdout = frame[frame["outer_role"].eq("holdout")].reset_index(drop=True)

    if len(dev) == 0:
        raise SystemExit("[holdout] No development cells found in split manifest.")
    if len(holdout) == 0:
        raise SystemExit("[holdout] No holdout cells found in split manifest.")

    print(f"[holdout] development: {dev['cell'].nunique()} cells, {len(dev)} anchors")
    print(f"[holdout] holdout:     {holdout['cell'].nunique()} cells, {len(holdout)} anchors")
    return dev, holdout


def _train_final_model(
    cfg: dict,
    dev: pd.DataFrame,
    best_epoch: int,
    pinn_seed: int,
    pinn_architecture: str,
    pinn_preprocessing: str,
    device: str,
    run_dir: Path,
) -> PINNInferenceSession:
    """Train NaPINN-Q on all development cells and return an inference session.

    No early stopping; epoch count is frozen by nested selection.
    Training the final model inside a caller-managed directory (not tempfile)
    so the checkpoint survives for downstream scripts.
    """
    from src.pinn.config_builder import build_trainer_config
    from src.pinn.dataset import build_anchor_dataset
    from src.pinn.losses import LossWeights
    from src.pinn.trainer import FoldPaths, FoldTrainingError, train_fold
    from src.pinn.utils import resolve_device

    pinn_cfg = cfg["pinn"]
    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    real_device = resolve_device(device)

    frame_raw = pd.read_parquet(features_dir / f"{pinn_preprocessing}.parquet")
    dataset = build_anchor_dataset(
        frame_raw, preprocessing=pinn_preprocessing,
        stress_cfg=pinn_cfg["stress"],
        audit_cfg=pinn_cfg.get("audit"),
        audit_path=None,
        features_dir=features_dir,
    )
    loss_weights = LossWeights.from_mapping(pinn_cfg["losses"])

    # All development cells → training; cv_fold=1 means "outer_train" when fold=0.
    # No outer_val cells: we evaluate manually after training.
    dev_cells = set(dev["cell"].astype(str).unique())
    anchor_frame = dataset.frame.copy()
    anchor_frame = anchor_frame[
        anchor_frame["cell"].astype(str).isin(dev_cells)
    ].reset_index(drop=True)
    anchor_frame["outer_role"] = "development"
    anchor_frame["cv_fold"] = 1  # all training; outer_val will be empty

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
        raise SystemExit(f"[holdout] Final model training failed: {exc}") from exc

    return PINNInferenceSession(run_dir, device=real_device), dataset


def _predict_split(
    session: PINNInferenceSession,
    frame: pd.DataFrame,
    dataset,
    split_label: str,
) -> pd.DataFrame:
    """Predict delta_Q in Ah for all rows in frame, return annotated DataFrame.

    ``frame`` must be a unified.parquet slice with anchor_id; it is joined to
    ``dataset.frame`` to get the PINN-required physics columns before inference.
    """
    anchor_frame = prepare_anchor_frame(frame, dataset)
    y_pred = session.predict_delta_q_ah(anchor_frame)
    y_true = frame[TARGET].to_numpy(dtype=float)

    out = frame[["cell", "condition"]].copy().reset_index(drop=True)
    if "anchor_id" in frame.columns:
        out["anchor_id"] = frame["anchor_id"].reset_index(drop=True).to_numpy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["error"] = y_pred - y_true
    out["abs_error"] = np.abs(y_pred - y_true)
    out["split"] = split_label
    out["model"] = "NaPINN-Q"
    out["evaluation_role"] = "exploratory_locked_holdout"
    return out


def _compute_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cell-macro and condition-macro MAE per split."""
    rows = []
    for split, grp in predictions.groupby("split"):
        cell_mae = grp.groupby("cell")["abs_error"].mean()
        cond_mae = grp.groupby("condition")["abs_error"].mean()
        rows.append({
            "split": split,
            "cell_macro_MAE_Ah": float(cell_mae.mean()),
            "condition_macro_MAE_Ah": float(cond_mae.mean()),
            "n_cells": int(cell_mae.shape[0]),
            "n_anchors": int(len(grp)),
        })
    return pd.DataFrame(rows)


def _load_classical_oof_for_bootstrap(
    cfg: dict,
    holdout_cells: set[str],
) -> pd.DataFrame | None:
    """Try to load classical per-anchor holdout predictions if available.

    Classical holdout predictions are written by src/pipeline/run_experiment.py
    if the holdout was already evaluated. If not available, returns None and the
    bootstrap step is skipped.
    """
    artifacts = Path(cfg["paths"]["artifacts"])
    # Classical holdout predictions: look for the per-anchor results file
    candidates = [
        artifacts / "results" / "classical_holdout_predictions.parquet",
        artifacts / "results" / "classical" / "holdout_predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            df["cell"] = df["cell"].astype(str)
            return df[df["cell"].isin(holdout_cells)].reset_index(drop=True)
    return None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="load data and print epoch decision, do not train")
    parser.add_argument("--pinn-seed", type=int, default=42,
                        help="primary seed (42 is the preregistered headline seed)")
    parser.add_argument("--pinn-architecture", type=str, default="NaPINN-Q")
    parser.add_argument("--pinn-preprocessing", type=str, default="unified")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    pinn_dir = _pinn_results_dir(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "pinn_holdout"
    )
    out.mkdir(parents=True, exist_ok=True)

    # Safeguard: refuse to overwrite completed holdout results.
    if (out / "pinn_holdout_predictions.parquet").exists() and not args.dry_run:
        raise SystemExit(
            f"[holdout] {out}/pinn_holdout_predictions.parquet already exists. "
            "This script is intended to run EXACTLY ONCE. If you need to rerun, "
            "manually delete the output directory first."
        )

    # Step 1: determine epoch from nested selection
    best_epoch, epoch_details = _read_nested_epoch(
        pinn_dir, args.pinn_architecture, args.pinn_preprocessing,
        args.outer_folds, args.pinn_seed,
    )
    (out / "pinn_holdout_epoch_decision.json").write_text(
        json.dumps(epoch_details, indent=2)
    )

    if args.dry_run:
        print("[holdout] --dry-run: epoch decision written, no training performed")
        return 0

    # Step 2: load development and holdout frames
    dev, holdout = _load_frames(cfg)

    # Step 3: train on all development cells
    model_dir = out / "final_model"
    model_dir.mkdir(exist_ok=True)
    print(f"[holdout] training final model ({best_epoch} epochs) on "
          f"{dev['cell'].nunique()} development cells ...")
    session, dataset = _train_final_model(
        cfg=cfg, dev=dev, best_epoch=best_epoch,
        pinn_seed=args.pinn_seed,
        pinn_architecture=args.pinn_architecture,
        pinn_preprocessing=args.pinn_preprocessing,
        device=args.device,
        run_dir=model_dir,
    )

    # Step 4: predict both splits (development fit check + holdout)
    dev_preds = _predict_split(session, dev, dataset, "development_fit")
    holdout_preds = _predict_split(session, holdout, dataset, "locked_holdout")
    all_preds = pd.concat([dev_preds, holdout_preds], ignore_index=True)
    all_preds.to_parquet(out / "pinn_holdout_predictions.parquet", index=False)

    # Step 5: metrics
    metrics = _compute_mae(all_preds)
    metrics.to_csv(out / "pinn_holdout_metrics.csv", index=False)
    print("\n[holdout] === Exploratory locked-holdout results ===")
    print(metrics.to_string(index=False))

    # Step 6: paired bootstrap vs classical (if classical holdout predictions exist)
    holdout_cells = set(holdout["cell"].astype(str).unique())
    classical = _load_classical_oof_for_bootstrap(cfg, holdout_cells)
    if classical is not None and len(classical) > 0:
        print(f"\n[holdout] running paired bootstrap vs "
              f"{classical['model'].nunique()} classical models ...")
        pinn_err = holdout_preds["error"].to_numpy(dtype=float)
        pinn_cells = holdout_preds["cell"].to_numpy()

        bootstrap_rows = []
        for model_name, grp in classical.groupby("model"):
            grp = grp.reset_index(drop=True)
            join_key = "anchor_id" if (
                "anchor_id" in holdout_preds.columns and "anchor_id" in grp.columns
            ) else "cell"
            merged = holdout_preds[["cell", join_key, "error"]].merge(
                grp[[join_key, "error"]].rename(columns={"error": "error_classical"}),
                on=join_key, how="inner",
                validate="one_to_one" if join_key == "anchor_id" else None,
            )
            if len(merged) == 0:
                continue
            result = paired_cell_bootstrap(
                errors_a=merged["error"].to_numpy(dtype=float),
                errors_b=merged["error_classical"].to_numpy(dtype=float),
                cells=merged["cell"].to_numpy(),
                n_bootstrap=args.n_bootstrap,
                seed=args.pinn_seed,
            )
            if result["ci_high"] < 0:
                interp = "NaPINN-Q significantly better"
            elif result["ci_low"] > 0:
                interp = f"{model_name} significantly better"
            else:
                interp = "statistically unresolved"
            bootstrap_rows.append({
                "reference": "NaPINN-Q",
                "comparator": model_name,
                "mean_paired_MAE_diff_Ah": result["mean_diff"],
                "ci_low_Ah": result["ci_low"],
                "ci_high_Ah": result["ci_high"],
                "p_reference_better": result["p_a_better"],
                "n_cells": result["n_cells"],
                "n_anchors": result["n_anchors"],
                "interpretation": interp,
                "evaluation_role": "exploratory_locked_holdout",
            })

        if bootstrap_rows:
            boot_df = pd.DataFrame(bootstrap_rows)
            boot_df.to_csv(out / "pinn_holdout_bootstrap.csv", index=False)
            print(boot_df.to_string(index=False))
    else:
        print("[holdout] no classical holdout predictions found; skipping bootstrap")
        print("  (run scripts/02_run_all_experiments.py with holdout enabled first)")

    print(f"\n[holdout] results labeled 'exploratory_locked_holdout'")
    print(f"[holdout] wrote {out}")
    return 0


def main() -> int:
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "pinn_holdout" / "logs"
    with run_log("18_run_final_pinn_holdout", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
