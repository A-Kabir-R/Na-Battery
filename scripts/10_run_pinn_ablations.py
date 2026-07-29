#!/usr/bin/env python3
"""Physics-loss ablation suite (A0..A5) using the winning development preprocessing.

Each ablation configuration is specified in ``config.yaml`` under
``pinn.ablation.configurations``. The winning preprocessing is selected by the
development-only CV ranking table produced by scripts/07.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.io.loaders import load_config  # noqa: E402
from src.pinn.dataset import apply_split_manifest, build_anchor_dataset  # noqa: E402
from src.pinn.evaluation import aggregate_predictions, target_metric_rows  # noqa: E402
from src.pinn.logging_utils import (  # noqa: E402
    get_logger, log_config_snapshot, log_dataset_summary, log_event, log_failure,
)
from src.pinn.losses import LossWeights  # noqa: E402
from src.pinn.trainer import FoldPaths, FoldTrainingError, TrainerConfig, train_fold  # noqa: E402
from src.pinn.utils import atomic_write_csv, git_commit, resolve_device  # noqa: E402
from src.splits.group_kfold import validate_locked_split_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Override the auto-selected preprocessing pipeline.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"])
    return parser.parse_args()


def _pinn_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / "results" / str(
        (cfg.get("pinn") or {}).get("results_subdir", "pinn_preprocessing_study")
    )


def _select_preprocessing(ranking_path: Path, default: str = "p2",
                          *, target: str = "next_rpt_Q_Ah") -> str:
    """Pick the winning preprocessing for a *specific* target.

    Sorting the ranking by pooled MAE across all targets is dominated by unit
    scale — capacity MAE in Ah is orders of magnitude smaller than SOH MAE in
    percentage points, so the first row would be a units artifact rather than
    a comparable model choice (Stage 4 diagnosis fix #25).
    """
    if not ranking_path.exists():
        return default
    ranking = pd.read_csv(ranking_path)
    if ranking.empty:
        return default
    focus = ranking
    if "architecture" in focus.columns:
        focus = focus[focus["architecture"] == "NaPINN-Q"]
    if "target" in focus.columns:
        focus = focus[focus["target"] == target]
    if "aggregation" in focus.columns:
        focus = focus[focus["aggregation"].astype(str).isin({"cell_macro", ""})]
    if "evaluation_role" in focus.columns:
        focus = focus[focus["evaluation_role"].astype(str).isin({"outer_validation", ""})]
    if focus.empty:
        return default
    focus = focus.sort_values("MAE") if "MAE" in focus.columns else focus
    winner = focus.iloc[0]["preprocessing"]
    return str(winner)


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    pinn_cfg = cfg["pinn"]
    ablation_cfg = pinn_cfg.get("ablation") or {}
    if not ablation_cfg.get("enabled", True):
        print("[pinn.ablation] disabled in config; nothing to do")
        return

    device = resolve_device(args.device)
    if args.device == "cuda" and device != "cuda":
        raise SystemExit("--device cuda requested but CUDA is not available")
    results = _pinn_dir(cfg)
    results.mkdir(parents=True, exist_ok=True)
    logger = get_logger("10_run_pinn_ablations", log_dir=results / "logs")
    log_event(logger, logging.INFO, "script_start",
              script="10_run_pinn_ablations",
              git_commit=git_commit(HERE.parent), args=vars(args))
    log_config_snapshot(logger, cfg)

    preprocessing = args.preprocessing or _select_preprocessing(
        results / "development_cv_model_ranking.csv", default="p2",
    )
    features_path = Path(cfg["paths"]["artifacts"]) / "features" / f"{preprocessing}.parquet"
    if not features_path.exists():
        raise SystemExit(f"feature file missing: {features_path}")
    frame = pd.read_parquet(features_path)
    reference = frame.dropna(subset=["next_rpt_Q_Ah"]).reset_index(drop=True)

    split_path = Path(cfg["paths"]["artifacts"]) / "splits" / "split_manifest.parquet"
    split_manifest = pd.read_parquet(split_path)
    validate_locked_split_manifest(
        reference, split_manifest,
        n_splits=int(cfg["split"]["n_splits"]),
        holdout_fraction=float(cfg["split"].get("holdout_fraction", 0.25)),
        random_state=int(cfg["split"]["random_state"]),
    )

    seeds = pinn_cfg["seeds"] if args.seed is None else [int(args.seed)]
    folds = list(range(int(cfg["split"]["n_splits"]))) if args.fold is None else [int(args.fold)]
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(pinn_cfg["training"]["maximum_epochs"])

    configurations = ablation_cfg["configurations"]
    plan = [
        (config_row, seed, fold)
        for config_row in configurations
        for seed in seeds
        for fold in folds
    ]

    print(f"[pinn.ablation] preprocessing: {preprocessing}")
    print(f"[pinn.ablation] configurations: {[c['name'] for c in configurations]}")
    print(f"[pinn.ablation] seeds: {seeds}")
    print(f"[pinn.ablation] folds: {folds}")
    print(f"[pinn.ablation] planned fits: {len(plan)}")

    dataset = build_anchor_dataset(
        frame, preprocessing=preprocessing, stress_cfg=pinn_cfg["stress"],
        audit_cfg=pinn_cfg.get("audit"),
        audit_path=results / f"temporal_feature_audit_{preprocessing}.csv",
        features_dir=features_path.parent,
    )
    attached = apply_split_manifest(dataset.frame, split_manifest)
    log_dataset_summary(logger, dataset, attached, preprocessing=preprocessing)

    prediction_paths = []
    failed_runs: list[dict[str, object]] = []
    for config_row, seed, fold in tqdm(plan, desc="[pinn.ablation]", unit="run"):
        arch = str(config_row["architecture"])
        weights = LossWeights.from_mapping(config_row["weights"])
        include_discrete = bool(config_row["weights"].get("discrete_state_transition", 0.0))
        # Per-ablation override for rate_uses_u_hat (A6 constrained rate).
        rate_uses_u_hat = bool(config_row.get(
            "rate_uses_u_hat", pinn_cfg["model"].get("rate_uses_u_hat", True)))
        run_dir = results / "ablation" / str(config_row["name"]) / preprocessing / f"fold_{fold}" / f"seed_{seed}"
        paths = FoldPaths(root=run_dir)
        trainer_cfg = TrainerConfig(
            architecture=arch, preprocessing=preprocessing, fold=fold, seed=seed,
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
            solution_activation=str(pinn_cfg["model"]["solution_activation"]),
            rate_activation=str(pinn_cfg["model"]["rate_activation"]),
            solution_dropout=float(pinn_cfg["model"].get("solution_dropout", 0.0)),
            rate_dropout=float(pinn_cfg["model"].get("rate_dropout", 0.0)),
            rate_uses_u_hat=rate_uses_u_hat,
            predict_delta_u=bool(pinn_cfg["model"].get("predict_delta_u", False)),
            batch_size=int(pinn_cfg["training"].get("batch_size", 0)),
            maximum_parameters=int(pinn_cfg["model"]["maximum_parameters"]),
            inner_split_seed=int(pinn_cfg.get("audit", {}).get(
                "inner_split_seed", 20240117)),
            two_phase_refit=bool(pinn_cfg["training"].get("two_phase_refit", True)),
            pde_gradient_min_norm=float(pinn_cfg["training"].get(
                "pde_gradient_min_norm", 1.0e-8)),
            pde_gradient_zero_patience=int(pinn_cfg["training"].get(
                "pde_gradient_zero_patience", 5)),
            use_gradient_balance=bool(pinn_cfg["training"].get(
                "use_gradient_balance", True)),
            log_every_epochs=int(pinn_cfg["logging"]["log_every_epochs"]),
            save_checkpoint_every_epochs=int(pinn_cfg["logging"]["save_checkpoint_every_epochs"]),
            log_gpu_memory=bool(pinn_cfg["logging"]["log_gpu_memory"]),
            log_gradient_norms=bool(pinn_cfg["logging"]["log_gradient_norms"]),
            device=device,
        )
        try:
            result = train_fold(
                dataset=dataset, frame=attached, config=trainer_cfg,
                fold_paths=paths, loss_weights=weights,
                u_max_source=str(pinn_cfg["bounds"]["u_max_source"]),
                u_max_constant=float(pinn_cfg["bounds"]["u_max_constant"]),
                u_max_margin=float(pinn_cfg["bounds"]["u_max_margin"]),
                tolerance_source=str(pinn_cfg["monotonicity"]["tolerance_source"]),
                tolerance_constant=float(pinn_cfg["monotonicity"]["tolerance_constant"]),
                tolerance_quantile=float(pinn_cfg["monotonicity"]["tolerance_quantile"]),
                force=args.force,
                ablation_name=str(config_row["name"]),
                include_discrete_transition=include_discrete,
                feature_file=features_path,
                split_manifest_file=split_path,
                git_commit_hash=str(git_commit(HERE.parent)),
                logger=logger,
            )
            prediction_paths.append(paths.predictions_path)
        except (FoldTrainingError, Exception) as exc:
            print(f"[pinn.ablation] FAILED {config_row['name']}|fold {fold}|seed {seed}: {exc}")
            log_failure(logger, "ablation_run_failed", exc,
                        ablation=str(config_row["name"]),
                        fold=fold, seed=seed, run_dir=str(run_dir))
            failed_runs.append({
                "ablation": str(config_row["name"]), "fold": fold, "seed": seed,
                "error_type": type(exc).__name__, "error_message": str(exc),
            })

    predictions = aggregate_predictions(prediction_paths)
    if predictions.empty:
        raise SystemExit("no ablation predictions produced")

    metric_frames = []
    # Group by evaluation_role too — otherwise training predictions bleed into
    # the validation MAE that drives A0..A6 comparison.
    group_columns = ["ablation", "architecture", "preprocessing", "fold", "seed",
                     "evaluation_role"]
    if "evaluation_role" not in predictions.columns:
        group_columns.remove("evaluation_role")
    groups = list(predictions.groupby(group_columns, dropna=False))
    for key, group in tqdm(groups, desc="[pinn.ablation] metrics", unit="run"):
        ablation = key[0]
        rows = target_metric_rows(group)
        rows["ablation"] = ablation
        metric_frames.append(rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    atomic_write_csv(metrics, results / "ablation_metrics.csv")

    atomic_write_csv(pd.DataFrame(failed_runs), results / "ablation_failed_runs.csv")
    print(f"[pinn.ablation] wrote {len(metrics)} metric rows; {len(failed_runs)} failed")
    log_event(logger, logging.INFO, "script_complete",
              metric_rows=len(metrics), predictions=len(prediction_paths),
              failed=len(failed_runs))
    if failed_runs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
