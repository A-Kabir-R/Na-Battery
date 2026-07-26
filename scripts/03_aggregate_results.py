"""Turn artifacts/results/raw_results.csv into aggregated + wide tables."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.evaluation.report import (
    aggregate,
    collect_predictions,
    overfitting_gap,
    pivot_comparison,
    prediction_metric_tables,
)
from src.io.loaders import load_config


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["artifacts"]) / "results"
    for optional_name in (
        "development_cv_model_ranking.csv", "feature_selection_frequency.csv",
        "feature_importance_raw.csv", "feature_importance_summary.csv",
    ):
        (results_dir / optional_name).unlink(missing_ok=True)
    raw_results = pd.read_csv(results_dir / "raw_results.csv")
    agg = aggregate(results_dir / "raw_results.csv")
    agg.to_csv(results_dir / "results_summary.csv", index=False)
    wide = pivot_comparison(agg)
    wide.to_csv(results_dir / "comparison_table.csv", index=False)
    gap = overfitting_gap(agg)
    gap.to_csv(results_dir / "overfitting_gap.csv", index=False)
    predictions = collect_predictions(
        Path(cfg["paths"]["artifacts"]), results_dir / "raw_results.csv"
    )
    predictions.to_parquet(results_dir / "all_predictions.parquet", index=False)
    evaluation = cfg.get("evaluation", {})
    prediction_metrics, group_metrics, horizon_metrics, plausibility = prediction_metric_tables(
        predictions,
        bootstrap_replicates=int(evaluation.get("bootstrap_replicates", 1000)),
        random_state=int(evaluation.get("bootstrap_random_state", 42)),
        horizon_edges=evaluation.get("horizon_bin_edges_days", [0, 1, 5, 15, 1e6]),
        horizon_labels=evaluation.get("horizon_bin_labels", ["<1d", "1-5d", "5-15d", ">15d"]),
    )
    prediction_metrics.to_csv(results_dir / "prediction_metrics.csv", index=False)
    group_metrics.to_csv(results_dir / "group_metrics.csv", index=False)
    horizon_metrics.to_csv(results_dir / "horizon_metrics.csv", index=False)
    plausibility.to_csv(results_dir / "prediction_plausibility.csv", index=False)

    if not prediction_metrics.empty:
        ranking = prediction_metrics[
            (prediction_metrics["evaluation_role"] == "cv_oof")
            & (prediction_metrics["aggregation"] == "cell_macro")
            & (prediction_metrics["metric"] == "MAE")
        ].sort_values(["target", "estimate"])
        ranking.to_csv(results_dir / "development_cv_model_ranking.csv", index=False)

    feature_rows = []
    artifacts = Path(cfg["paths"]["artifacts"])

    def _current_completed_partition(path: Path) -> bool:
        relative = path.relative_to(artifacts / "folds")
        preprocessing, target, model, partition = relative.parts[:4]
        fold = -1 if partition == "holdout" else int(partition.removeprefix("fold_"))
        status_path = path.parent / "fold_status.json"
        if not status_path.exists():
            return False
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rows = raw_results[
            raw_results["preprocessing"].astype(str).eq(preprocessing)
            & raw_results["target"].astype(str).eq(target)
            & raw_results["model"].astype(str).eq(model)
            & raw_results["fold"].eq(fold)
            & raw_results["status"].eq("completed")
        ]
        return (
            status.get("status") == "completed"
            and status.get("fingerprint") in set(rows["fingerprint"].dropna())
        )

    for path in sorted((artifacts / "folds").glob("*/*/*/*/selected_features.json")):
        if not _current_completed_partition(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(artifacts / "folds")
        preprocessing, target, model, partition = relative.parts[:4]
        role = "holdout_final" if partition == "holdout" else "cv"
        for feature in payload.get("features", []):
            feature_rows.append({
                "preprocessing": preprocessing, "target": target, "model": model,
                "partition": partition, "role": role, "feature": feature,
            })
    feature_frame = pd.DataFrame(feature_rows)
    if not feature_frame.empty:
        feature_frequency = (
            feature_frame.groupby(["preprocessing", "target", "model", "role", "feature"])
            .size().rename("selection_count").reset_index()
        )
        feature_frequency.to_csv(results_dir / "feature_selection_frequency.csv", index=False)

    importance_frames = []
    for path in sorted((artifacts / "folds").glob("*/*/*/*/feature_importance.parquet")):
        if not _current_completed_partition(path):
            continue
        importance_frames.append(pd.read_parquet(path))
    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True)
        importance.to_csv(results_dir / "feature_importance_raw.csv", index=False)
        importance_summary = (
            importance.groupby([
                "preprocessing", "target", "model", "evaluation_role",
                "importance_type", "feature",
            ])["normalized_importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        importance_summary.to_csv(results_dir / "feature_importance_summary.csv", index=False)
    print(f"[agg] wrote results_summary.csv  ({len(agg)} rows)")
    print(f"[agg] wrote comparison_table.csv ({len(wide)} rows)")
    print(f"[agg] wrote overfitting_gap.csv  ({len(gap)} rows)")
    print(f"[agg] wrote prediction_metrics.csv ({len(prediction_metrics)} rows)")
    print(f"[agg] wrote group_metrics.csv ({len(group_metrics)} rows)")
    print(f"[agg] wrote horizon_metrics.csv ({len(horizon_metrics)} rows)")
    print(wide.head(20).to_string())


if __name__ == "__main__":
    main()
