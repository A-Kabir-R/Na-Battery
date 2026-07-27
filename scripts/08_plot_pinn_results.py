#!/usr/bin/env python3
"""Render PINN publication plots into the results directory."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import logging  # noqa: E402

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.io.loaders import load_config  # noqa: E402
from src.pinn.logging_utils import get_logger, log_event  # noqa: E402
from src.pinn.plotting import render_all_pinn_plots  # noqa: E402
from src.pinn.utils import git_commit  # noqa: E402


def _results_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / "results" / str(
        (cfg.get("pinn") or {}).get("results_subdir", "pinn_preprocessing_study")
    )


def _safe_read(path: Path, parquet: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if parquet else pd.read_csv(path)


def main() -> None:
    cfg = load_config()
    results = _results_dir(cfg)
    plots_dir = results / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("08_plot_pinn_results", log_dir=results / "logs")
    log_event(logger, logging.INFO, "script_start",
              script="08_plot_pinn_results", git_commit=git_commit(HERE.parent))

    stages = [
        ("all_predictions.parquet", "predictions", True),
        ("epoch_logs_combined.parquet", "epoch_log", True),
        ("prediction_metrics.csv", "metrics", False),
        ("ablation_metrics.csv", "ablation", False),
        ("model_complexity.csv", "complexity", False),
    ]
    loaded: dict[str, pd.DataFrame] = {}
    for filename, key, parquet in tqdm(stages, desc="[pinn.plot] load", unit="artifact"):
        loaded[key] = _safe_read(results / filename, parquet=parquet)
        log_event(logger, logging.INFO, "artifact_loaded", key=key,
                  path=str(results / filename), rows=int(len(loaded[key])))

    # Optional combined-family metrics from the classical+PINN report — used
    # only for the accuracy comparison bar; skipped silently if not present.
    combined_metrics_path = (Path(cfg["paths"]["artifacts"]) / "results"
                              / "combined_classical_pinn"
                              / "prediction_metrics_combined.csv")
    combined_metrics = _safe_read(combined_metrics_path) if combined_metrics_path.exists() else None

    render_all_pinn_plots(
        epoch_log=loaded["epoch_log"],
        predictions=loaded["predictions"],
        target_metrics=loaded["metrics"],
        ablation_metrics=loaded["ablation"],
        complexity=loaded["complexity"],
        combined_metrics=combined_metrics,
        output_dir=plots_dir,
    )
    print(f"[pinn.plot] rendered plots into {plots_dir}")
    log_event(logger, logging.INFO, "script_complete", plots_dir=str(plots_dir))


if __name__ == "__main__":
    main()
