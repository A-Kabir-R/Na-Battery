#!/usr/bin/env python3
"""Run every DNN-Q and NaPINN-Q training fold for the configured seeds.

The classical pipeline (scripts/02_run_all_experiments.py) is untouched: this
script writes to ``artifacts/results/pinn_preprocessing_study/`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.io.loaders import load_config  # noqa: E402
from src.pinn.dataset import (  # noqa: E402
    apply_split_manifest, build_anchor_dataset,
)
from src.pinn.logging_utils import (  # noqa: E402
    get_logger, log_config_snapshot, log_dataset_summary, log_event, log_failure,
)
from src.pinn.losses import LossWeights  # noqa: E402
from src.pinn.trainer import FoldPaths, FoldTrainingError, TrainerConfig, train_fold  # noqa: E402
from src.pinn.utils import (  # noqa: E402
    atomic_write_csv, atomic_write_json, git_commit, resolve_device,
)
from src.splits.group_kfold import validate_locked_split_manifest  # noqa: E402
import logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only one fold/seed/preprocessing/architecture for a few epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--preprocessing", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--allow-partial", action="store_true",
                        help="Exit 0 even if some runs failed (default: exit 2 on any failure).")
    return parser.parse_args()


def _results_dir(cfg: dict) -> Path:
    base = Path(cfg["paths"]["artifacts"]) / "results"
    subdir = (cfg.get("pinn") or {}).get("results_subdir") or "pinn_preprocessing_study"
    return base / str(subdir)


def _load_split(cfg: dict, reference_frame: pd.DataFrame) -> pd.DataFrame:
    split_path = Path(cfg["paths"]["artifacts"]) / "splits" / "split_manifest.parquet"
    if not split_path.exists():
        raise FileNotFoundError(
            f"split manifest missing at {split_path}; run scripts/02_run_all_experiments.py "
            "first so the classical and PINN pipelines share identical folds."
        )
    split_manifest = pd.read_parquet(split_path)
    validate_locked_split_manifest(
        reference_frame, split_manifest,
        n_splits=int(cfg["split"]["n_splits"]),
        holdout_fraction=float(cfg["split"].get("holdout_fraction", 0.25)),
        random_state=int(cfg["split"]["random_state"]),
    )
    return split_manifest


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    pinn_cfg = cfg["pinn"]
    # Device selection is independent of AMP (Stage 3 diagnosis fix #20).
    # Priority: --device flag > SIB_DEVICE env var > "cuda" (with graceful
    # cpu fallback). If --device cuda is passed explicitly, refuse silent
    # cpu fallback so a GPU smoke test cannot accidentally succeed on cpu.
    import os as _os
    device_request = args.device or _os.environ.get("SIB_DEVICE")
    device = resolve_device(device_request)
    if args.device == "cuda" and device != "cuda":
        raise SystemExit("--device cuda requested but CUDA is not available")
    results_dir = _results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "epoch_logs").mkdir(parents=True, exist_ok=True)
    logger = get_logger("06_run_pinn_experiments", log_dir=results_dir / "logs")
    log_event(logger, logging.INFO, "script_start",
              script="06_run_pinn_experiments", git_commit=git_commit(HERE.parent),
              args=vars(args))
    log_config_snapshot(logger, cfg)

    preprocessing_choices = pinn_cfg["preprocessing"]
    architectures = pinn_cfg["architectures"]
    seeds = pinn_cfg["seeds"]
    folds = list(range(int(cfg["split"]["n_splits"])))
    if args.preprocessing:
        preprocessing_choices = [args.preprocessing]
    if args.architecture:
        architectures = [args.architecture]
    if args.seed is not None:
        seeds = [int(args.seed)]
    if args.fold is not None:
        folds = [int(args.fold)]
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(pinn_cfg["training"]["maximum_epochs"])
    if args.smoke_test:
        max_epochs = min(5, max_epochs)
        preprocessing_choices = preprocessing_choices[:1]
        architectures = architectures[:1]
        seeds = seeds[:1]
        folds = folds[:1]

    plan = list(product(architectures, preprocessing_choices, folds, seeds))

    print(f"[pinn] model name: {pinn_cfg['model_name']}")
    print(f"[pinn] architectures: {', '.join(architectures)}")
    print(f"[pinn] preprocessing: {', '.join(preprocessing_choices)}")
    print(f"[pinn] unique configurations: {len(architectures) * len(preprocessing_choices)}")
    print(f"[pinn] outer folds: {len(folds)}")
    print(f"[pinn] development seeds: {len(seeds)}")
    print(f"[pinn] planned outer-fold fits: {len(plan)}")
    print(f"[pinn] fundamental predicted state: normalized next-RPT capacity")
    print(f"[pinn] derived target views: 4")
    print(f"[pinn] autograd derivative: enabled")
    print(f"[pinn] collocation points: enabled")
    print(f"[pinn] points per transition: {pinn_cfg['collocation']['points_per_transition']}")
    print(f"[pinn] PDE residual: du/ds + r")
    print(f"[pinn] holdout access during screening: disabled")
    print(f"[pinn] device: {device}")
    print(f"[pinn] results directory: {results_dir}")

    if args.dry_run:
        for arch, prep, fold, seed in plan:
            print(f"[pinn]   {arch:>10} | {prep:>3} | fold={fold} | seed={seed}")
        return

    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    p2 = pd.read_parquet(features_dir / "p2.parquet")
    reference = p2.dropna(subset=["next_rpt_Q_Ah"]).reset_index(drop=True)
    split_manifest = _load_split(cfg, reference)

    manifest_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    total_start = time.time()

    plan_bar = tqdm(plan, desc="[pinn] runs", unit="run")
    for arch, prep, fold, seed in plan_bar:
        plan_bar.set_postfix_str(f"{arch}|{prep}|f{fold}|s{seed}")
        run_dir = results_dir / "runs" / f"{arch}" / prep / f"fold_{fold}" / f"seed_{seed}"
        paths = FoldPaths(root=run_dir)

        try:
            frame = pd.read_parquet(features_dir / f"{prep}.parquet")
            dataset = build_anchor_dataset(
                frame, preprocessing=prep, stress_cfg=pinn_cfg["stress"],
                audit_cfg=pinn_cfg.get("audit"),
                audit_path=results_dir / f"temporal_feature_audit_{prep}.csv",
                features_dir=features_dir,
            )
            attached = apply_split_manifest(dataset.frame, split_manifest)
            log_dataset_summary(logger, dataset, attached, preprocessing=prep)
            trainer_cfg = TrainerConfig(
                architecture=arch, preprocessing=prep, fold=fold, seed=seed,
                max_epochs=max_epochs,
                min_epochs=int(pinn_cfg["training"]["minimum_epochs"]),
                early_stopping_patience=int(pinn_cfg["training"]["early_stopping_patience"]),
                learning_rate=float(pinn_cfg["training"]["learning_rate"]),
                weight_decay=float(pinn_cfg["training"]["weight_decay"]),
                scheduler_factor=float(pinn_cfg["training"]["scheduler_factor"]),
                scheduler_patience=int(pinn_cfg["training"]["scheduler_patience"]),
                minimum_learning_rate=float(pinn_cfg["training"]["minimum_learning_rate"]),
                gradient_clip_norm=float(pinn_cfg["training"]["gradient_clip_norm"]),
                use_amp=bool(pinn_cfg["training"]["use_amp"]),
                physics_float32=bool(pinn_cfg["training"]["physics_float32"]),
                curriculum_warmup_fraction=float(pinn_cfg["curriculum"]["warmup_fraction"]),
                curriculum_ramp_end_fraction=float(pinn_cfg["curriculum"]["ramp_end_fraction"]),
                huber_delta=float(pinn_cfg["losses"]["huber_delta"]),
                collocation_points=int(pinn_cfg["collocation"]["points_per_transition"]),
                quadrature_method=str(pinn_cfg["quadrature"]["method"]),
                quadrature_nodes=int(pinn_cfg["quadrature"]["nodes"]),
                solution_hidden_dims=tuple(int(h) for h in pinn_cfg["model"]["solution_hidden_dims"]),
                rate_hidden_dims=tuple(int(h) for h in pinn_cfg["model"]["rate_hidden_dims"]),
                dnn_solution_hidden_dims=tuple(int(h) for h in pinn_cfg["model"].get(
                    "dnn_solution_hidden_dims", pinn_cfg["model"]["solution_hidden_dims"])),
                solution_activation=str(pinn_cfg["model"]["solution_activation"]),
                rate_activation=str(pinn_cfg["model"]["rate_activation"]),
                rate_uses_u_hat=bool(pinn_cfg["model"].get("rate_uses_u_hat", True)),
                maximum_parameters=int(pinn_cfg["model"]["maximum_parameters"]),
                inner_split_seed=int(pinn_cfg.get("audit", {}).get(
                    "inner_split_seed", 20240117)),
                two_phase_refit=bool(pinn_cfg["training"].get("two_phase_refit", True)),
                pde_gradient_min_norm=float(pinn_cfg["training"].get(
                    "pde_gradient_min_norm", 1.0e-8)),
                pde_gradient_zero_patience=int(pinn_cfg["training"].get(
                    "pde_gradient_zero_patience", 5)),
                log_every_epochs=int(pinn_cfg["logging"]["log_every_epochs"]),
                save_checkpoint_every_epochs=int(pinn_cfg["logging"]["save_checkpoint_every_epochs"]),
                log_gpu_memory=bool(pinn_cfg["logging"]["log_gpu_memory"]),
                log_gradient_norms=bool(pinn_cfg["logging"]["log_gradient_norms"]),
                device=device,
            )
            weights = LossWeights.from_mapping(pinn_cfg["losses"])
            result = train_fold(
                dataset=dataset, frame=attached, config=trainer_cfg,
                fold_paths=paths, loss_weights=weights,
                u_max_source=str(pinn_cfg["bounds"]["u_max_source"]),
                u_max_constant=float(pinn_cfg["bounds"]["u_max_constant"]),
                u_max_margin=float(pinn_cfg["bounds"]["u_max_margin"]),
                tolerance_source=str(pinn_cfg["monotonicity"]["tolerance_source"]),
                tolerance_constant=float(pinn_cfg["monotonicity"]["tolerance_constant"]),
                tolerance_quantile=float(pinn_cfg["monotonicity"]["tolerance_quantile"]),
                resume=args.resume, force=args.force,
                dry_run=args.dry_run,
                # Fingerprint feeds — file hashes + git commit so a data
                # refresh invalidates cached completed runs.
                feature_file=features_dir / f"{prep}.parquet",
                split_manifest_file=Path(cfg["paths"]["artifacts"]) / "splits" / "split_manifest.parquet",
                git_commit_hash=str(git_commit(HERE.parent)),
                logger=logger,
            )
            manifest_rows.append({
                "architecture": arch, "preprocessing": prep, "fold": fold, "seed": seed,
                "status": result.status, "best_epoch": result.best_epoch,
                "best_validation_mae": result.best_validation_mae,
                "trainable_parameters": result.trainable_parameters,
                "predictions_path": str(result.predictions_path),
                "run_dir": str(run_dir),
                "git_commit": git_commit(HERE.parent),
            })
        except (FoldTrainingError, Exception) as exc:  # pragma: no cover
            failed_rows.append({
                "architecture": arch, "preprocessing": prep, "fold": fold, "seed": seed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "run_dir": str(run_dir),
                "git_commit": git_commit(HERE.parent),
                "timestamp": time.time(),
            })
            manifest_rows.append({
                "architecture": arch, "preprocessing": prep, "fold": fold, "seed": seed,
                "status": "failed",
                "best_epoch": -1,
                "best_validation_mae": float("nan"),
                "trainable_parameters": 0,
                "predictions_path": "",
                "run_dir": str(run_dir),
                "git_commit": git_commit(HERE.parent),
            })
            tqdm.write(f"[pinn] FAILED {arch}|{prep}|f{fold}|s{seed}: {exc}")
            log_failure(logger, "run_failed", exc,
                        architecture=arch, preprocessing=prep,
                        fold=fold, seed=seed, run_dir=str(run_dir))

    manifest_df = pd.DataFrame(manifest_rows)
    atomic_write_csv(manifest_df, results_dir / "experiment_manifest.csv")
    atomic_write_csv(pd.DataFrame(failed_rows), results_dir / "failed_runs.csv")
    atomic_write_json({"total_runs": len(plan),
                       "completed": int((manifest_df["status"] == "completed").sum()) if not manifest_df.empty else 0,
                       "failed": len(failed_rows),
                       "elapsed_seconds": time.time() - total_start,
                       "git_commit": git_commit(HERE.parent)},
                      results_dir / "run_summary.json")
    print(f"[pinn] wrote experiment manifest: {len(manifest_df)} rows")
    print(f"[pinn] failed runs: {len(failed_rows)}")
    print(f"[pinn] elapsed: {time.time() - total_start:.1f}s")
    completed_count = int((manifest_df["status"] == "completed").sum()) if not manifest_df.empty else 0
    log_event(logger, logging.INFO, "script_complete",
              total_runs=len(plan),
              completed=completed_count,
              failed=len(failed_rows),
              elapsed_seconds=time.time() - total_start)

    # Exit non-zero when any run failed so the pipeline halts before
    # aggregation contaminates results (Stage 4 diagnosis fix #32).
    if failed_rows and not args.allow_partial:
        print(f"[pinn] exiting non-zero: {len(failed_rows)} failed run(s); "
              f"pass --allow-partial to proceed anyway")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
