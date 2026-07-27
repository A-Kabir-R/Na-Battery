#!/usr/bin/env python3
"""Aggregate every PINN fold prediction into publication-ready tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.io.loaders import load_config  # noqa: E402
from src.pinn.evaluation import (  # noqa: E402
    aggregate_predictions, horizon_metric_rows, physics_metric_rows,
    target_metric_rows,
)
from src.pinn.logging_utils import get_logger, log_event  # noqa: E402
from src.pinn.utils import atomic_write_csv, atomic_write_parquet, git_commit  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Aggregate even when the manifest is incomplete "
                             "(diagnostic only — do not use for publication).")
    parser.add_argument("--include-failed-logs", action="store_true",
                        help="Include epoch logs from failed/unknown-status runs "
                             "in the combined epoch table.")
    return parser.parse_args()


def _results_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / "results" / str(
        (cfg.get("pinn") or {}).get("results_subdir", "pinn_preprocessing_study")
    )


def _iter_prediction_paths(run_root: Path):
    """Yield only ``predictions.parquet`` files that live alongside a
    ``status.json`` with ``status == "completed"``.

    Stage 3/4 diagnosis fix #21: never let failed or partial runs enter the
    aggregation. A run without ``status.json`` is treated as incomplete.
    """
    for predictions in sorted(run_root.rglob("predictions.parquet")):
        status_path = predictions.parent / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            continue
        if status.get("status") != "completed":
            continue
        yield predictions


def _iter_epoch_logs(run_root: Path, *, include_failed: bool = False):
    for epoch_log in sorted(run_root.rglob("epoch_log.parquet")):
        status_path = epoch_log.parent.parent / "status.json"
        if not status_path.exists():
            if include_failed:
                yield epoch_log
            continue
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            if include_failed:
                yield epoch_log
            continue
        if status.get("status") != "completed":
            if include_failed:
                yield epoch_log
            continue
        yield epoch_log


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    results = _results_dir(cfg)
    run_root = results / "runs"
    logger = get_logger("07_aggregate_pinn_results", log_dir=results / "logs")
    log_event(logger, logging.INFO, "script_start",
              script="07_aggregate_pinn_results",
              git_commit=git_commit(HERE.parent), run_root=str(run_root),
              allow_partial=args.allow_partial,
              include_failed_logs=args.include_failed_logs)

    if not run_root.exists():
        raise SystemExit(f"no run directory at {run_root}; run scripts/06 first")

    # Cross-check against the experiment manifest to detect silently missing
    # runs — the aggregator must not produce a ranking table from a partial
    # sweep (Stage 4 diagnosis fix #21).
    manifest_path = results / "experiment_manifest.csv"
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        expected_completed = int((manifest_df["status"] == "completed").sum()) \
            if "status" in manifest_df.columns else int(len(manifest_df))
        planned = int(len(manifest_df))
    else:
        manifest_df = pd.DataFrame()
        expected_completed = 0
        planned = 0

    prediction_paths = list(_iter_prediction_paths(run_root))
    log_event(logger, logging.INFO, "prediction_paths_discovered",
              n=len(prediction_paths), expected_completed=expected_completed,
              planned=planned)
    if expected_completed and len(prediction_paths) < expected_completed:
        msg = (f"[pinn.aggregate] found {len(prediction_paths)} completed "
               f"prediction files but the manifest expected {expected_completed}")
        if args.allow_partial:
            print(f"{msg}; --allow-partial set, continuing with diagnostic aggregation.")
            log_event(logger, logging.WARNING, "partial_aggregation_allowed",
                      found=len(prediction_paths), expected=expected_completed)
        else:
            raise SystemExit(f"{msg}; refusing to aggregate. Pass --allow-partial "
                             f"for diagnostic aggregation.")
    if planned and expected_completed != planned and not args.allow_partial:
        raise SystemExit(
            f"[pinn.aggregate] experiment manifest lists {planned} planned runs "
            f"but only {expected_completed} completed; refuse to aggregate a "
            f"partial sweep. Pass --allow-partial for diagnostic aggregation."
        )
    predictions_frames = []
    for path in tqdm(prediction_paths, desc="[pinn.aggregate] loading", unit="run"):
        try:
            predictions_frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.exception("failed_to_read_predictions", extra={
                "structured": {"path": str(path), "error": str(exc)}
            })
    predictions = pd.concat(predictions_frames, ignore_index=True) \
        if predictions_frames else pd.DataFrame()
    if predictions.empty:
        raise SystemExit("no PINN predictions found; nothing to aggregate")

    atomic_write_parquet(predictions, results / "all_predictions.parquet")
    log_event(logger, logging.INFO, "predictions_written",
              n_rows=len(predictions),
              path=str(results / "all_predictions.parquet"))

    metric_rows = []
    physics_rows = []
    horizon_rows = []
    complexity_rows = []
    # Group by evaluation_role too — otherwise inner_train, inner_validation
    # and outer_validation rows collapse into a single metric row per run and
    # inflate reported validation accuracy (post-Stage-4 correction).
    group_columns = ["architecture", "preprocessing", "fold", "seed", "evaluation_role"]
    if "evaluation_role" not in predictions.columns:
        group_columns.remove("evaluation_role")
    groups = list(predictions.groupby(group_columns, dropna=False))
    seen_complexity: set[tuple[object, ...]] = set()
    for key, group in tqdm(groups, desc="[pinn.aggregate] metrics", unit="run"):
        metric_rows.append(target_metric_rows(group))
        physics_rows.append(physics_metric_rows(group))
        horizon_rows.append(horizon_metric_rows(group))
        # First four positions are always arch/prep/fold/seed — evaluation_role
        # is optional, but complexity is per (arch,prep,fold,seed) regardless.
        arch, prep, fold, seed = key[0], key[1], key[2], key[3]
        complexity_key = (arch, prep, fold, seed)
        if complexity_key in seen_complexity:
            continue
        seen_complexity.add(complexity_key)
        run_dir = run_root / str(arch) / str(prep) / f"fold_{fold}" / f"seed_{seed}"
        status_path = run_dir / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                complexity_rows.append({
                    "architecture": arch, "preprocessing": prep, "fold": fold,
                    "seed": seed,
                    "trainable_parameters": status.get("trainable_parameters", np.nan),
                    "best_epoch": status.get("best_epoch", np.nan),
                    "best_validation_mae": status.get("best_validation_mae", np.nan),
                    "status": status.get("status", "unknown"),
                })
            except Exception:
                pass

    metrics = pd.concat([m for m in metric_rows if not m.empty], ignore_index=True) \
        if metric_rows else pd.DataFrame()
    physics = pd.concat([m for m in physics_rows if not m.empty], ignore_index=True) \
        if physics_rows else pd.DataFrame()
    horizons = pd.concat([m for m in horizon_rows if not m.empty], ignore_index=True) \
        if horizon_rows else pd.DataFrame()

    atomic_write_csv(metrics, results / "prediction_metrics.csv")
    atomic_write_csv(physics, results / "physics_metrics.csv")
    atomic_write_csv(horizons, results / "horizon_metrics.csv")
    atomic_write_csv(pd.DataFrame(complexity_rows), results / "model_complexity.csv")

    # Raw long-format results
    long_rows = []
    for _, row in metrics.iterrows():
        for metric_name in ("MAE", "RMSE", "R2", "MaxError", "MAPE"):
            long_rows.append({
                "model_family": "physics_informed_neural_network" if row["architecture"] == "NaPINN-Q"
                    else "data_driven_neural_network",
                "architecture": row["architecture"],
                "preprocessing": row["preprocessing"],
                "target": row["target"],
                "aggregation": row["aggregation"],
                "fold": row.get("fold", ""),
                "seed": row.get("seed", ""),
                "evaluation_role": row.get("evaluation_role", ""),
                "metric": metric_name,
                "value": row.get(metric_name, np.nan),
                "n": row.get("n", 0),
            })
    raw = pd.DataFrame(long_rows)
    atomic_write_csv(raw, results / "raw_results.csv")

    # Development-only CV ranking table
    dev = metrics[metrics.get("evaluation_role", "").astype(str) == "outer_validation"] \
        if "evaluation_role" in metrics.columns else pd.DataFrame()
    if not dev.empty:
        ranking = (
            dev[dev["aggregation"] == "cell_macro"]
            .groupby(["architecture", "preprocessing", "target"])["MAE"]
            .mean().reset_index()
            .sort_values("MAE")
        )
        atomic_write_csv(ranking, results / "development_cv_model_ranking.csv")

    epoch_paths = list(_iter_epoch_logs(run_root,
                                         include_failed=args.include_failed_logs))
    if epoch_paths:
        epoch_frames = []
        for path in tqdm(epoch_paths, desc="[pinn.aggregate] epoch logs", unit="run"):
            try:
                epoch_frames.append(pd.read_parquet(path))
            except Exception as exc:
                logger.exception("failed_to_read_epoch_log",
                                 extra={"structured": {"path": str(path),
                                                        "error": str(exc)}})
        if epoch_frames:
            combined_epoch = pd.concat(epoch_frames, ignore_index=True)
            atomic_write_parquet(combined_epoch, results / "epoch_logs_combined.parquet")
            log_event(logger, logging.INFO, "epoch_logs_combined",
                      n_rows=len(combined_epoch))

    print(f"[pinn.aggregate] wrote metrics: {len(metrics)} rows, "
          f"physics: {len(physics)} rows, horizons: {len(horizons)} rows")
    log_event(logger, logging.INFO, "script_complete",
              metric_rows=len(metrics), physics_rows=len(physics),
              horizon_rows=len(horizons), complexity_rows=len(complexity_rows))


if __name__ == "__main__":
    main()
