"""Render publication tables as raster and vector figures."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src.evaluation.metrics import regression_metrics
from src.io.loaders import load_config

LOWER_IS_BETTER = {"MAE", "RMSE", "MAPE", "MaxError"}


def _safe(value: object) -> str:
    return "".join(character if str(character).isalnum() or character in "._-" else "_"
                   for character in str(value))


def _role_label(role: str) -> str:
    return "exploratory locked holdout" if role == "holdout" else role


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(out: Path, stem: str, formats: list[str], dpi: int) -> None:
    plt.tight_layout()
    for extension in formats:
        plt.savefig(out / f"{_safe(stem)}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close()


def _metric_heatmaps(metrics: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    groups = metrics.groupby(["target", "evaluation_role", "aggregation", "metric"])
    for (target, role, aggregation, metric), group in tqdm(
        list(groups), desc="[plot] metric heatmaps", unit="figure"
    ):
        pivot = group.pivot_table(index="model", columns="preprocessing", values="estimate")
        if pivot.empty:
            continue
        plt.figure(figsize=(7, max(4, 0.36 * len(pivot))))
        cmap = "RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn"
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap=cmap, center=0 if metric == "R2" else None)
        plt.title(f"{target}: {metric} | {_role_label(role)} | {aggregation}")
        plt.xlabel("Feature pipeline")
        plt.ylabel("Model")
        _save(out, f"heatmap_{target}_{metric}_{role}_{aggregation}", formats, dpi)


def _r2_forests(metrics: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    r2 = metrics[metrics["metric"] == "R2"].copy()
    for (target, role, aggregation), group in r2.groupby(
        ["target", "evaluation_role", "aggregation"]
    ):
        group = group.sort_values("estimate")
        labels = group["model"] + " [" + group["preprocessing"] + "]"
        y = np.arange(len(group))
        low = (group["estimate"] - group["ci_low"]).clip(lower=0).fillna(0)
        high = (group["ci_high"] - group["estimate"]).clip(lower=0).fillna(0)
        plt.figure(figsize=(9, max(5, 0.28 * len(group))))
        plt.errorbar(group["estimate"], y, xerr=np.vstack([low, high]), fmt="o", capsize=2)
        plt.yticks(y, labels)
        plt.axvline(0, color="black", lw=0.8, ls="--")
        plt.axvline(1, color="black", lw=0.5, ls=":")
        plt.xlabel("R² (95% cell-bootstrap CI)")
        plt.title(f"R² model comparison: {target} | {_role_label(role)} | {aggregation}")
        plt.grid(axis="x", alpha=0.25)
        _save(out, f"r2_forest_{target}_{role}_{aggregation}", formats, dpi)


def _parity_and_residual_plots(predictions: pd.DataFrame, out: Path,
                               formats: list[str], dpi: int) -> None:
    keys = ["preprocessing", "target", "model", "evaluation_role"]
    groups = list(predictions[predictions["evaluation_role"].isin(["cv_oof", "holdout"])].groupby(keys))
    for key, group in tqdm(groups, desc="[plot] parity/residual", unit="model-role"):
        preprocessing, target, model, role = key
        values = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        finite = np.isfinite(group["y_true"]) & np.isfinite(group["y_pred"])
        plot = group[finite]
        if plot.empty:
            continue
        minimum = float(min(plot["y_true"].min(), plot["y_pred"].min()))
        maximum = float(max(plot["y_true"].max(), plot["y_pred"].max()))
        padding = max((maximum - minimum) * 0.05, 1e-6)
        plt.figure(figsize=(6.2, 5.7))
        sns.scatterplot(data=plot, x="y_true", y="y_pred", hue="condition", s=45, alpha=0.8)
        plt.plot([minimum - padding, maximum + padding], [minimum - padding, maximum + padding],
                 color="black", ls="--", lw=1, label="Ideal")
        plt.xlim(minimum - padding, maximum + padding)
        plt.ylim(minimum - padding, maximum + padding)
        plt.xlabel(f"Observed {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{model} [{preprocessing}] | {_role_label(role)}\n"
                  f"R²={values['R2']:.3f}, MAE={values['MAE']:.4g}, "
                  f"n={len(plot)} rows/{plot['cell'].nunique()} cells")
        plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        stem = f"parity_{target}_{role}_{preprocessing}_{model}"
        _save(out, stem, formats, dpi)

        residual = plot["y_pred"] - plot["y_true"]
        plt.figure(figsize=(6.2, 5.0))
        plt.scatter(plot["y_pred"], residual, c=plot["next_rpt_horizon_days"]
                    if "next_rpt_horizon_days" in plot else "tab:blue", cmap="viridis", alpha=0.8)
        plt.axhline(0, color="black", ls="--", lw=1)
        plt.xlabel(f"Predicted {target}")
        plt.ylabel("Residual (predicted - observed)")
        plt.title(f"Residuals: {model} [{preprocessing}] | {_role_label(role)}")
        if "next_rpt_horizon_days" in plot:
            plt.colorbar(label="Forecast horizon (days)")
        plt.grid(alpha=0.2)
        _save(out, f"residual_{target}_{role}_{preprocessing}_{model}", formats, dpi)


def _fold_r2(raw: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    frame = raw[(raw["split"] == "cv_validation") & (raw["metric"] == "R2")].copy()
    for target, group in frame.groupby("target"):
        group["model_pipeline"] = group["model"] + " [" + group["preprocessing"] + "]"
        order = group.groupby("model_pipeline")["value"].median().sort_values().index
        plt.figure(figsize=(12, max(6, 0.25 * len(order))))
        sns.boxplot(data=group, y="model_pipeline", x="value", order=order, color="#9ecae1")
        sns.stripplot(data=group, y="model_pipeline", x="value", order=order,
                      color="black", size=3, alpha=0.7)
        plt.axvline(0, color="black", ls="--", lw=0.8)
        plt.xlabel("Development CV fold R²")
        plt.ylabel("")
        plt.title(f"Fold-to-fold R² stability: {target}")
        _save(out, f"r2_fold_distribution_{target}", formats, dpi)


def _condition_r2(group_metrics: pd.DataFrame, out: Path,
                  formats: list[str], dpi: int) -> None:
    frame = group_metrics[
        (group_metrics["group_type"] == "condition") & (group_metrics["metric"] == "R2")
    ]
    for (target, role, preprocessing), group in frame.groupby(
        ["target", "evaluation_role", "preprocessing"]
    ):
        pivot = group.pivot_table(index="model", columns="group_value", values="estimate")
        if pivot.empty:
            continue
        plt.figure(figsize=(max(10, 0.65 * len(pivot.columns)), max(5, 0.35 * len(pivot))))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".2f")
        plt.title(f"Per-condition R²: {target} | {_role_label(role)} | {preprocessing}")
        plt.xlabel("Condition")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha="right")
        _save(out, f"r2_condition_{target}_{role}_{preprocessing}", formats, dpi)


def _cell_r2_distributions(group_metrics: pd.DataFrame, out: Path,
                           formats: list[str], dpi: int) -> None:
    frame = group_metrics[
        (group_metrics["group_type"] == "cell")
        & (group_metrics["metric"] == "R2")
        & np.isfinite(group_metrics["estimate"])
    ].copy()
    for (target, role), group in frame.groupby(["target", "evaluation_role"]):
        group["model_pipeline"] = group["model"] + " [" + group["preprocessing"] + "]"
        order = group.groupby("model_pipeline")["estimate"].median().sort_values().index
        plt.figure(figsize=(11, max(6, 0.27 * len(order))))
        sns.boxplot(data=group, y="model_pipeline", x="estimate", order=order, color="#a1d99b")
        sns.stripplot(data=group, y="model_pipeline", x="estimate", order=order,
                      color="black", size=2.5, alpha=0.55)
        plt.axvline(0, color="black", ls="--", lw=0.8)
        plt.xlabel("Per-cell R²")
        plt.ylabel("")
        plt.title(f"Per-cell R² distribution: {target} | {_role_label(role)}")
        _save(out, f"r2_cell_distribution_{target}_{role}", formats, dpi)


def _r2_fit_evaluation(raw: pd.DataFrame, out: Path,
                       formats: list[str], dpi: int) -> None:
    r2 = raw[raw["metric"] == "R2"].copy()
    for fit_split, evaluation_split, label in (
        ("cv_train", "cv_validation", "development_cv"),
        ("development_fit", "holdout", "locked_holdout"),
    ):
        subset = r2[r2["split"].isin([fit_split, evaluation_split])]
        if subset.empty:
            continue
        means = subset.groupby(
            ["preprocessing", "target", "model", "split"]
        )["value"].mean().unstack("split").dropna(subset=[fit_split, evaluation_split])
        for target, group in means.reset_index().groupby("target"):
            plt.figure(figsize=(7, 6))
            sns.scatterplot(data=group, x=fit_split, y=evaluation_split,
                            hue="model", style="preprocessing", s=60)
            low = min(group[fit_split].min(), group[evaluation_split].min())
            high = max(group[fit_split].max(), group[evaluation_split].max())
            plt.plot([low, high], [low, high], color="black", ls="--", lw=0.8)
            plt.axhline(0, color="grey", lw=0.5)
            plt.xlabel(f"{fit_split} R²")
            plt.ylabel(f"{evaluation_split} R²")
            qualifier = "fold-mean stability diagnostic" if label == "development_cv" else "exploratory holdout"
            plt.title(f"R² generalization ({qualifier}): {target}")
            plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            _save(out, f"r2_fit_vs_evaluation_{target}_{label}", formats, dpi)


def _horizon_plots(horizon: pd.DataFrame, out: Path,
                   formats: list[str], dpi: int) -> None:
    if horizon.empty:
        return
    order = ["<1d", "1-5d", "5-15d", ">15d"]
    for (target, role, aggregation, metric), group in horizon.groupby(
        ["target", "evaluation_role", "aggregation", "metric"]
    ):
        if metric not in {"R2", "MAE", "RMSE"}:
            continue
        group = group.copy()
        group["horizon_bin"] = pd.Categorical(group["horizon_bin"], order, ordered=True)
        plt.figure(figsize=(11, 5.5))
        sns.lineplot(data=group.sort_values("horizon_bin"), x="horizon_bin", y="estimate",
                     hue="model", style="preprocessing", markers=True, dashes=False)
        if metric == "R2":
            plt.axhline(0, color="black", ls="--", lw=0.8)
        plt.xlabel("Forecast horizon")
        plt.ylabel(metric)
        plt.title(f"{metric} by forecast horizon: {target} | {_role_label(role)} | {aggregation}")
        plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(alpha=0.2)
        _save(out, f"horizon_{metric}_{target}_{role}_{aggregation}", formats, dpi)


def _plausibility_plot(plausibility: pd.DataFrame, out: Path,
                       formats: list[str], dpi: int) -> None:
    for (target, role), group in plausibility.groupby(["target", "evaluation_role"]):
        group = group.copy()
        group["model_pipeline"] = group["model"] + " [" + group["preprocessing"] + "]"
        group = group.sort_values("invalid_prediction_fraction")
        plt.figure(figsize=(10, max(5, 0.25 * len(group))))
        sns.barplot(data=group, y="model_pipeline", x="invalid_prediction_fraction", color="#d95f0e")
        plt.xlabel("Fraction of physically invalid predictions")
        plt.ylabel("")
        plt.title(f"Prediction plausibility: {target} | {_role_label(role)}")
        _save(out, f"prediction_plausibility_{target}_{role}", formats, dpi)


def _feature_importance_plots(results: Path, out: Path,
                              formats: list[str], dpi: int) -> None:
    path = results / "feature_importance_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    final = frame[frame["evaluation_role"] == "holdout_final"]
    for (target, preprocessing, model), group in final.groupby(
        ["target", "preprocessing", "model"]
    ):
        group = group.nlargest(20, "mean").sort_values("mean")
        if group.empty:
            continue
        plt.figure(figsize=(8, max(4.5, 0.28 * len(group))))
        plt.barh(group["feature"], group["mean"], xerr=group["std"].fillna(0),
                 color="#3182bd", alpha=0.85)
        plt.xlabel("Normalized model importance")
        plt.title(f"Feature importance: {target} | {preprocessing} | {model}")
        plt.grid(axis="x", alpha=0.2)
        _save(out, f"feature_importance_{target}_{preprocessing}_{model}", formats, dpi)


def _loss_curves(curves_dir: Path, out: Path, formats: list[str], dpi: int,
                 raw_results: pd.DataFrame) -> None:
    files = sorted(curves_dir.glob("*.parquet")) if curves_dir.exists() else []
    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"[plot] failed to read {path.name}: {exc}")
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    if not raw_results.empty:
        current = set(map(tuple, raw_results[[
            "preprocessing", "target", "model",
        ]].drop_duplicates().astype(str).to_numpy()))
        data = data[data.apply(
            lambda row: (str(row["preprocessing"]), str(row["target"]), str(row["model"]))
            in current,
            axis=1,
        )]
    for (preprocessing, target, model), group in tqdm(
        list(data.groupby(["preprocessing", "target", "model"])),
        desc="[plot] loss curves", unit="figure",
    ):
        stats = group.groupby(["iteration", "split"])["loss"].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(7, 4.5))
        for split, split_data in stats.groupby("split"):
            split_data = split_data.sort_values("iteration")
            plt.plot(split_data["iteration"], split_data["mean"], label=split, lw=1.5)
            if split_data["std"].notna().any():
                plt.fill_between(
                    split_data["iteration"], split_data["mean"] - split_data["std"],
                    split_data["mean"] + split_data["std"], alpha=0.2,
                )
        plt.xlabel("Iteration")
        plt.ylabel(group["loss_name"].iloc[0])
        plt.title(f"{preprocessing} | {target} | {model}")
        plt.legend()
        plt.grid(alpha=0.3)
        _save(out, f"loss_curve_{preprocessing}_{target}_{model}", formats, dpi)


def _dataset_plots(artifacts: Path, out: Path, formats: list[str], dpi: int) -> None:
    features = artifacts / "features"
    eligibility_path = features / "target_eligibility.parquet"
    if eligibility_path.exists():
        eligibility = pd.read_parquet(eligibility_path)
        counts = eligibility.groupby(["condition", "supervised_target_available"]).size().unstack(fill_value=0)
        counts.columns = ["Unavailable" if not column else "Available" for column in counts.columns]
        counts.plot(kind="bar", stacked=True, figsize=(12, 5.5), color=["#d95f02", "#1b9e77"])
        plt.ylabel("Cells")
        plt.title("Supervised target availability by cycling condition")
        plt.xticks(rotation=45, ha="right")
        _save(out, "cohort_target_availability", formats, dpi)

    anchor_path = features / "anchor_samples.parquet"
    if anchor_path.exists():
        anchors = pd.read_parquet(anchor_path)
        eligible = anchors[anchors["next_rpt_Q_Ah"].notna()]
        for column, label in (
            ("next_rpt_Q_Ah", "Next RPT capacity (Ah)"),
            ("next_rpt_SOH_pct", "Next RPT SOH (%)"),
            ("delta_next_rpt_Q_Ah", "Next-minus-previous RPT capacity (Ah)"),
            ("delta_next_rpt_SOH_pct", "Next-minus-previous RPT SOH (%)"),
            ("next_rpt_horizon_days", "Forecast horizon (days)"),
        ):
            if column not in eligible:
                continue
            plt.figure(figsize=(7, 4.8))
            sns.histplot(eligible[column].dropna(), kde=True, bins=25)
            plt.xlabel(label)
            plt.ylabel("Anchors")
            plt.title(f"Distribution of {label.lower()}")
            _save(out, f"distribution_{column}", formats, dpi)

    rpt_path = features / "cu_capacity.parquet"
    if rpt_path.exists():
        rpt = pd.read_parquet(rpt_path)
        valid = rpt[rpt["rpt_qc_status"] == "valid"].copy()
        if not valid.empty:
            valid["file_start_time"] = pd.to_datetime(valid["file_start_time"])
            valid = valid.sort_values(["condition", "cell", "file_start_time", "test_id"])
            baseline = valid[valid["visit"] == 0].groupby(["condition", "cell"])[
                "reference_capacity_Ah"
            ].first().rename("initial_Q")
            valid = valid.join(baseline, on=["condition", "cell"])
            valid["RPT_SOH_pct"] = 100 * valid["reference_capacity_Ah"] / valid["initial_Q"]
            plt.figure(figsize=(11, 6))
            sns.lineplot(data=valid, x="visit", y="RPT_SOH_pct", units="cell",
                         estimator=None, hue="condition", alpha=0.65)
            plt.axhline(80, color="black", ls="--", lw=1, label="80% EOL")
            plt.ylabel("RPT SOH (%)")
            plt.title("Cell-level RPT SOH trajectories")
            plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            _save(out, "rpt_soh_trajectories_all_cells", formats, dpi)

    degradation_path = artifacts / "results" / "rpt_cell_degradation.csv"
    if degradation_path.exists():
        degradation = pd.read_csv(degradation_path)
        for column, label in (
            ("minimum_rpt_SOH_pct", "Minimum observed RPT SOH (%)"),
            ("soh_slope_pct_per_100_days", "SOH slope (% per 100 days)"),
            ("soh_slope_pct_per_100_cycles", "SOH slope (% per 100 cycles)"),
            ("soh_slope_pct_per_100_EFC", "SOH slope (% per 100 EFC)"),
            ("projected_day_at_80pct", "Exploratory projected day at 80% SOH"),
            ("projected_cycle_at_80pct", "Exploratory projected cycle at 80% SOH"),
            ("projected_EFC_at_80pct", "Exploratory projected EFC at 80% SOH"),
        ):
            values = degradation.dropna(subset=[column])
            if values.empty:
                continue
            plt.figure(figsize=(12, 5.5))
            sns.boxplot(data=values, x="condition", y=column, color="#9ecae1")
            sns.stripplot(data=values, x="condition", y=column, color="black", size=3)
            plt.ylabel(label)
            plt.xlabel("Cycling condition")
            plt.title(f"{label} by condition")
            plt.xticks(rotation=45, ha="right")
            _save(out, f"degradation_{column}_by_condition", formats, dpi)


def _results_dir(cfg: dict) -> Path:
    base = Path(cfg["paths"]["artifacts"]) / "results"
    subdir = (cfg.get("experiment") or {}).get("results_subdir") or ""
    subdir = str(subdir).strip()
    return base / subdir if subdir else base


def main() -> None:
    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    results = _results_dir(cfg)
    print(f"[plot] results directory: {results}", flush=True)
    out = results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plotting = cfg.get("plotting", {})
    formats = ["png"]  # PNG-only per publication figure spec.
    dpi = int(plotting.get("dpi", 300))
    for extension in formats:
        for stale in out.glob(f"*.{extension}"):
            stale.unlink()
    sns.set_theme(style="whitegrid", context="paper")

    metrics = _read_csv(results / "prediction_metrics.csv")
    prediction_path = results / "all_predictions.parquet"
    predictions = pd.read_parquet(prediction_path) if prediction_path.exists() else pd.DataFrame()
    group_metrics = _read_csv(results / "group_metrics.csv")

    # Pruned figure set: only the plots required by the publication spec.
    if not metrics.empty:
        _r2_forests(metrics, out, formats, dpi)
    if not predictions.empty:
        _parity_and_residual_plots(predictions, out, formats, dpi)
    if not group_metrics.empty:
        _condition_r2(group_metrics, out, formats, dpi)
    # Retain the per-cell SOH / capacity trajectory figures from _dataset_plots.
    _dataset_plots(artifacts, out, formats, dpi)
    # Loss / LR curves → supplementary material only.
    supp = out / "supplementary"
    supp.mkdir(parents=True, exist_ok=True)
    _loss_curves(artifacts / "loss_curves", supp, formats, dpi, _read_csv(results / "raw_results.csv"))
    print(f"[plot] wrote publication figures to {out}")


if __name__ == "__main__":
    main()
