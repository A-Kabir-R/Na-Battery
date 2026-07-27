#!/usr/bin/env python3
"""Combine classical and PINN result tables into a single publication view.

Stage-4 diagnosis fixes applied here:

* Classical rows are normalized so that ``architecture`` never remains
  ``NaN`` — its value falls back to ``model`` (issue #22).
* ``--require-classical`` refuses to write publication tables when the
  classical raw results are missing (issue #23).
* Predictions are combined into ``all_predictions_combined.parquet`` and
  aggregate metrics into ``prediction_metrics_combined.csv`` — the
  previously-misnamed ``prediction_metrics_combined.parquet`` is no
  longer written (issue #24).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.io.loaders import load_config  # noqa: E402
from src.pinn.logging_utils import get_logger, log_event  # noqa: E402
from src.pinn.utils import atomic_write_csv, atomic_write_parquet, git_commit  # noqa: E402

CLASSICAL_FAMILY_MAP = {
    "DummyMean": "statistical_baseline",
    "PreviousRPT": "persistence_baseline",
    "ExtraTrees": "classical_machine_learning",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-classical", action="store_true",
                        help="Exit non-zero when classical raw_results.csv is missing.")
    parser.add_argument("--require-pinn", action="store_true",
                        help="Exit non-zero when the PINN raw_results.csv is missing.")
    return parser.parse_args()


def _classical_dir(cfg: dict) -> Path:
    subdir = (cfg.get("experiment") or {}).get("results_subdir") or ""
    return Path(cfg["paths"]["artifacts"]) / "results" / (subdir or "")


def _pinn_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / "results" / str(
        (cfg.get("pinn") or {}).get("results_subdir", "pinn_preprocessing_study")
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _normalize_classical(classical: pd.DataFrame) -> pd.DataFrame:
    """Ensure classical rows carry a valid ``architecture`` column.

    Pre-fix behaviour: after concatenating with PINN rows, classical rows
    could end up with ``NaN`` architecture and disappear from grouped tables
    because pandas excludes NaN group keys.
    """
    if classical.empty:
        return classical
    out = classical.copy()
    if "model" in out.columns and "architecture" not in out.columns:
        out["architecture"] = out["model"]
    elif "architecture" in out.columns and "model" in out.columns:
        out["architecture"] = out["architecture"].fillna(out["model"])
    elif "architecture" not in out.columns:
        out["architecture"] = "unknown"
    return out


def _classify_family(row: pd.Series) -> str:
    if "model_family" in row and pd.notna(row.get("model_family")):
        return str(row["model_family"])
    architecture = str(row.get("architecture", "") or row.get("model", ""))
    if architecture in CLASSICAL_FAMILY_MAP:
        return CLASSICAL_FAMILY_MAP[architecture]
    if architecture == "NaPINN-Q":
        return "physics_informed_neural_network"
    if architecture == "DNN-Q":
        return "data_driven_neural_network"
    return "unknown"


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    combined_dir = Path(cfg["paths"]["artifacts"]) / "results" / "combined_classical_pinn"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "plots").mkdir(parents=True, exist_ok=True)
    logger = get_logger("09_combine_classical_pinn_results",
                         log_dir=combined_dir / "logs")
    log_event(logger, logging.INFO, "script_start",
              script="09_combine_classical_pinn_results",
              git_commit=git_commit(HERE.parent), args=vars(args))

    classical_dir = _classical_dir(cfg)
    pinn_dir = _pinn_dir(cfg)

    stages = [
        ("classical_raw", classical_dir / "raw_results.csv"),
        ("pinn_raw", pinn_dir / "raw_results.csv"),
    ]
    loaded: dict[str, pd.DataFrame] = {}
    for key, path in tqdm(stages, desc="[combine] load raw", unit="table"):
        loaded[key] = _read_csv(path)
        log_event(logger, logging.INFO, "raw_table_loaded",
                  key=key, path=str(path), rows=int(len(loaded[key])))
    classical_raw = _normalize_classical(loaded["classical_raw"])
    pinn_raw = loaded["pinn_raw"]

    if classical_raw.empty:
        msg = f"classical raw_results.csv missing at {classical_dir}"
        if args.require_classical:
            log_event(logger, logging.ERROR, "missing_classical", path=str(classical_dir))
            raise SystemExit(f"[combine] --require-classical set but {msg}")
        print(f"[combine] WARNING: {msg}; combined report will contain PINN rows only.")
    if pinn_raw.empty:
        msg = f"PINN raw_results.csv missing at {pinn_dir}"
        if args.require_pinn:
            log_event(logger, logging.ERROR, "missing_pinn", path=str(pinn_dir))
            raise SystemExit(f"[combine] --require-pinn set but {msg}")
        print(f"[combine] WARNING: {msg}; combined report will contain classical rows only.")

    combined_raw = pd.concat([classical_raw, pinn_raw], ignore_index=True)
    if combined_raw.empty:
        raise SystemExit("no result rows to combine")

    combined_raw["model_family"] = combined_raw.apply(_classify_family, axis=1)

    atomic_write_csv(combined_raw, combined_dir / "raw_results_combined.csv")

    # Publication main table: cell-macro MAE by architecture × preprocessing × target.
    if {"metric", "value", "architecture", "preprocessing", "target"}.issubset(combined_raw.columns):
        main_mae = combined_raw[combined_raw["metric"] == "MAE"]
        if "aggregation" in main_mae.columns:
            main_mae = main_mae[main_mae["aggregation"].fillna("").isin(["cell_macro", ""])]
        pub_table = (
            main_mae.groupby(["model_family", "architecture", "preprocessing", "target"],
                             dropna=False)
            ["value"].mean().reset_index()
        )
        atomic_write_csv(pub_table, combined_dir / "publication_main_table.csv")

    # Combined predictions (classical predictions live under folds/, so PINN
    # only for now; the classical pipeline can hook in later by writing a
    # matching parquet under classical_dir / "all_predictions.parquet").
    prediction_frames = []
    classical_predictions = _read_parquet(classical_dir / "all_predictions.parquet")
    pinn_predictions = _read_parquet(pinn_dir / "all_predictions.parquet")
    if not classical_predictions.empty:
        prediction_frames.append(classical_predictions)
    if not pinn_predictions.empty:
        prediction_frames.append(pinn_predictions)
    if prediction_frames:
        combined_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
        atomic_write_parquet(combined_predictions,
                             combined_dir / "all_predictions_combined.parquet")

    # Prediction metrics table combined (per architecture/preprocessing/target).
    if not combined_raw.empty and {"metric", "value"}.issubset(combined_raw.columns):
        metrics_wide = (
            combined_raw
            .groupby(["model_family", "architecture", "preprocessing", "target",
                      "aggregation", "metric"], dropna=False)
            ["value"].mean().reset_index()
        )
        atomic_write_csv(metrics_wide, combined_dir / "prediction_metrics_combined.csv")

    # Physics table
    physics = _read_csv(pinn_dir / "physics_metrics.csv")
    if not physics.empty:
        atomic_write_csv(physics, combined_dir / "publication_physics_table.csv")

    # Ablation
    ablation = _read_csv(pinn_dir / "ablation_metrics.csv")
    if not ablation.empty:
        atomic_write_csv(ablation, combined_dir / "publication_ablation_table.csv")

    # Complexity
    complexity = _read_csv(pinn_dir / "model_complexity.csv")
    if not complexity.empty:
        atomic_write_csv(complexity, combined_dir / "publication_complexity_table.csv")

    # Development ranking
    ranking = _read_csv(pinn_dir / "development_cv_model_ranking.csv")
    if not ranking.empty:
        atomic_write_csv(ranking, combined_dir / "development_ranking_combined.csv")

    print(f"[combine] wrote combined report into {combined_dir}")
    log_event(logger, logging.INFO, "script_complete",
              combined_dir=str(combined_dir),
              classical_rows=int(len(classical_raw)),
              pinn_rows=int(len(pinn_raw)))


if __name__ == "__main__":
    main()
