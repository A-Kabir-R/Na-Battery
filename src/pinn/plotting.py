"""Publication-quality plots for NaPINN-Q.

Every plot is written as PNG (300 dpi) plus PDF. Missing data raises a warning
but does not crash the pipeline — combined reports must remain resilient to
partial runs.

Stage-4 diagnosis fixes applied here:

* Performance figures (parity, residuals, per-cell trajectories, degradation
  rate, PDE residual) filter to ``evaluation_role == "outer_validation"``
  by default (issues #26, #27).
* Per-cell trajectories use ``stress_target`` as the x-coordinate — the
  target-time coordinate — rather than ``stress_current`` (issue #26).
* Seed-stability boxplot restricts to a single target, aggregation and
  evaluation role so one scalar per seed is compared (issue #28).
* Ablation bar aggregates per-ablation before plotting to survive duplicate
  index labels from multi-fold / multi-seed rows (issue #29).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


FORMATS = ("png",)
DEFAULT_ROLE = "outer_validation"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _skip(name: str, reason: str) -> None:
    warnings.warn(f"[pinn.plot] skipping {name}: {reason}", RuntimeWarning)


def _outer_only(predictions: pd.DataFrame) -> pd.DataFrame:
    if "evaluation_role" not in predictions.columns:
        return predictions
    return predictions[predictions["evaluation_role"].astype(str).eq(DEFAULT_ROLE)]


def _epoch_curve(epoch_log: pd.DataFrame, y_column: str, title: str, path: Path,
                 log_y: bool = False) -> None:
    if epoch_log.empty or y_column not in epoch_log.columns:
        _skip(y_column, "column missing"); return
    fig, ax = plt.subplots(figsize=(7, 4))
    for (arch, prep, fold, seed), group in epoch_log.groupby(
            ["architecture", "preprocessing", "fold", "seed"], dropna=False):
        ax.plot(group["epoch"], group[y_column],
                label=f"{arch}|{prep}|f{fold}|s{seed}", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_column)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    if epoch_log["architecture"].nunique() * epoch_log["fold"].nunique() < 12:
        ax.legend(fontsize=6, loc="best")
    _save(fig, path)


def plot_epoch_curves(epoch_log: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if epoch_log.empty:
        _skip("epoch_curves", "epoch log is empty"); return
    for column, title, log_y in [
        ("train_total_loss", "Total training loss", True),
        ("train_data_loss", "Data loss", True),
        ("train_pde_loss", "PDE loss", True),
        ("train_pde_anchor_loss", "PDE loss (anchor)", True),
        ("train_pde_collocation_loss", "PDE loss (collocation)", True),
        ("train_initial_condition_loss", "Initial-condition loss", True),
        ("train_integral_loss", "Integral consistency loss", True),
        ("train_monotonicity_loss", "Monotonicity loss", True),
        ("train_bounds_loss", "Bounds loss", True),
        ("train_rate_loss", "Rate regularization loss", True),
        ("validation_MAE", "Inner-validation MAE", False),
        ("validation_RMSE", "Inner-validation RMSE", False),
        ("validation_R2", "Inner-validation R²", False),
        ("learning_rate", "Learning rate", True),
        ("gradient_norm_total", "Total gradient norm", True),
        ("gradient_norm_pde", "PDE-component gradient norm", True),
        ("GPU_memory_allocated_MB", "GPU memory allocated (MB)", False),
    ]:
        _epoch_curve(epoch_log, column, title, output_dir / f"epoch_{column}", log_y=log_y)


def plot_predicted_vs_true(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    for target, true_col, pred_col in [
        ("capacity", "true_next_Q_Ah", "predicted_next_Q_Ah"),
        ("soh", "true_next_SOH_pct", "predicted_next_SOH_pct"),
    ]:
        if subset_all.empty or true_col not in subset_all.columns:
            _skip(f"pred_vs_true_{target}", "columns missing or no outer rows"); continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(subset_all[true_col], subset_all[pred_col], s=6, alpha=0.5)
        finite = np.isfinite(subset_all[true_col]) & np.isfinite(subset_all[pred_col])
        if finite.any():
            lo = float(min(subset_all[true_col][finite].min(),
                           subset_all[pred_col][finite].min()))
            hi = float(max(subset_all[true_col][finite].max(),
                           subset_all[pred_col][finite].max()))
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(true_col)
        ax.set_ylabel(pred_col)
        ax.set_title(f"Predicted vs observed ({target}, outer validation)")
        _save(fig, output_dir / f"predicted_vs_true_{target}")


def plot_residual_distribution(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    for target, true_col, pred_col in [
        ("capacity", "true_next_Q_Ah", "predicted_next_Q_Ah"),
        ("soh", "true_next_SOH_pct", "predicted_next_SOH_pct"),
    ]:
        if subset_all.empty or true_col not in subset_all.columns:
            _skip(f"residual_{target}", "columns missing or no outer rows"); continue
        residual = (subset_all[pred_col] - subset_all[true_col]).astype(float).dropna()
        if residual.empty:
            _skip(f"residual_{target}", "no finite residuals"); continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(residual, bins=40, alpha=0.75)
        ax.set_xlabel(f"residual ({target})")
        ax.set_ylabel("count")
        ax.set_title(f"Residual distribution ({target}, outer validation)")
        _save(fig, output_dir / f"residual_hist_{target}")


def plot_per_cell_trajectories(predictions: pd.DataFrame, output_dir: Path,
                                target_col: str = "predicted_next_Q_Ah",
                                true_col: str = "true_next_Q_Ah") -> None:
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    if subset_all.empty or true_col not in subset_all.columns:
        _skip("per_cell_trajectories", "columns missing or no outer rows"); return
    if "stress_target" not in subset_all.columns:
        _skip("per_cell_trajectories", "stress_target column missing"); return
    cells = sorted(subset_all["cell_id"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for cell in cells[:30]:
        sub = subset_all[subset_all["cell_id"].astype(str).eq(cell)].sort_values("stress_target")
        ax.plot(sub["stress_target"], sub[true_col], color="black", alpha=0.4, linewidth=0.6)
        ax.plot(sub["stress_target"], sub[target_col], color="tab:red", alpha=0.4, linewidth=0.6)
    ax.set_xlabel("stress at target time (physical units)")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Per-cell observed (black) vs predicted (red) capacity, outer validation")
    _save(fig, output_dir / "per_cell_trajectories")


def plot_degradation_rate(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    for suffix, column, xlabel in [
        ("normalized", "predicted_degradation_rate_normalized",
         "predicted rate (normalized-stress units)"),
        ("physical", "predicted_degradation_rate_physical",
         "predicted rate (physical stress units)"),
        # legacy fallback
        ("legacy", "predicted_degradation_rate",
         "predicted degradation rate"),
    ]:
        if subset_all.empty or column not in subset_all.columns:
            continue
        values = pd.to_numeric(subset_all[column], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(values, bins=50, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(f"Predicted degradation-rate distribution ({suffix})")
        _save(fig, output_dir / f"degradation_rate_distribution_{suffix}")


def plot_pde_residual(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    for suffix, column, xlabel in [
        ("normalized", "pde_residual_normalized",
         "PDE residual (normalized-stress units)"),
        ("physical", "pde_residual_physical",
         "PDE residual (physical stress units)"),
        ("legacy", "pde_residual", "PDE residual (du/ds + r)"),
    ]:
        if subset_all.empty or column not in subset_all.columns:
            continue
        values = pd.to_numeric(subset_all[column], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(values, bins=50, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(f"Governing-equation residual distribution ({suffix})")
        _save(fig, output_dir / f"pde_residual_distribution_{suffix}")


def plot_ablation_bar(ablation_metrics: pd.DataFrame, output_dir: Path,
                      metric: str = "MAE") -> None:
    """Per-ablation mean of a scalar metric.

    Aggregates across folds and seeds *before* plotting so that duplicate
    index labels (multiple rows per ablation) cannot break the reindex step.
    Restricts to ``target == 'next_rpt_Q_Ah'``, ``aggregation == 'cell_macro'``
    and ``evaluation_role == 'outer_validation'`` when those columns exist.
    """
    output_dir = Path(output_dir)
    if ablation_metrics.empty or metric not in ablation_metrics.columns:
        _skip("ablation_bar", "columns missing"); return
    subset = ablation_metrics
    if "target" in subset.columns:
        subset = subset[subset["target"] == "next_rpt_Q_Ah"]
    if "aggregation" in subset.columns:
        subset = subset[subset["aggregation"].astype(str).isin({"cell_macro", ""})]
    if "evaluation_role" in subset.columns:
        subset = subset[subset["evaluation_role"].astype(str).isin({DEFAULT_ROLE, ""})]
    if subset.empty:
        _skip("ablation_bar", "no rows after filters"); return
    if "ablation" not in subset.columns:
        _skip("ablation_bar", "no ablation column"); return
    summary = (
        subset.groupby("ablation", as_index=False, dropna=False)[metric]
        .mean()
        .sort_values(metric)
    )
    if summary.empty:
        _skip("ablation_bar", "empty summary"); return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["ablation"].astype(str), summary[metric])
    ax.set_ylabel(f"mean {metric} (outer validation, cell macro)")
    ax.set_title(f"Physics ablation ({metric})")
    plt.xticks(rotation=30, ha="right")
    _save(fig, output_dir / f"ablation_{metric.lower()}")


def plot_seed_stability(target_metrics: pd.DataFrame, output_dir: Path,
                         metric: str = "MAE") -> None:
    output_dir = Path(output_dir)
    if target_metrics.empty or "seed" not in target_metrics.columns:
        _skip("seed_stability", "column missing"); return
    subset = target_metrics
    if "target" in subset.columns:
        subset = subset[subset["target"] == "next_rpt_Q_Ah"]
    if "aggregation" in subset.columns:
        subset = subset[subset["aggregation"].astype(str) == "cell_macro"]
    if "evaluation_role" in subset.columns:
        subset = subset[subset["evaluation_role"].astype(str) == DEFAULT_ROLE]
    if subset.empty:
        _skip("seed_stability", "no rows after filters"); return
    per_seed = subset.groupby(
        ["architecture", "seed"], as_index=False, dropna=False
    )[metric].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [g[metric].dropna().to_numpy() for _, g in per_seed.groupby("architecture")]
    labels = [name for name, _ in per_seed.groupby("architecture")]
    if not groups:
        _skip("seed_stability", "no groups"); return
    ax.boxplot(groups, labels=labels)
    ax.set_ylabel(f"{metric} (per-seed average across folds)")
    ax.set_title(f"Seed stability ({metric}, outer validation, cell macro)")
    _save(fig, output_dir / f"seed_stability_{metric.lower()}")


def plot_model_complexity(complexity: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of trainable parameters by architecture (Stage 4 fix #11 dashboard)."""
    output_dir = Path(output_dir)
    if complexity.empty or "trainable_parameters" not in complexity.columns:
        _skip("model_complexity", "columns missing"); return
    summary = (
        complexity.groupby("architecture", as_index=False, dropna=False)
        ["trainable_parameters"].mean()
        .sort_values("trainable_parameters", ascending=False)
    )
    if summary.empty:
        _skip("model_complexity", "no rows"); return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(summary["architecture"].astype(str), summary["trainable_parameters"])
    ax.set_ylabel("trainable parameters (avg over folds/seeds)")
    ax.set_title("Model complexity")
    plt.xticks(rotation=15, ha="right")
    _save(fig, output_dir / "model_complexity")


def plot_pde_residual_vs_stress(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of PDE residual vs stress_target (Stage 4 fix #30)."""
    output_dir = Path(output_dir)
    subset_all = _outer_only(predictions)
    column = "pde_residual_physical" if "pde_residual_physical" in subset_all.columns \
        else ("pde_residual_normalized" if "pde_residual_normalized" in subset_all.columns
              else "pde_residual")
    if subset_all.empty or column not in subset_all.columns or "stress_target" not in subset_all.columns:
        _skip("pde_residual_vs_stress", "columns missing"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(subset_all["stress_target"], pd.to_numeric(subset_all[column], errors="coerce"),
               s=6, alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("stress at target time")
    ax.set_ylabel(column)
    ax.set_title("PDE residual vs stress (outer validation)")
    _save(fig, output_dir / "pde_residual_vs_stress")


def plot_residuals_by_condition(predictions: pd.DataFrame, output_dir: Path,
                                 *, true_col: str = "true_next_Q_Ah",
                                 pred_col: str = "predicted_next_Q_Ah") -> None:
    """Boxplot of |residual| binned by temperature, DOD and C-rate.

    Produces three panels — temperature, depth-of-discharge, discharge C-rate —
    so heterogeneity of error by operating condition is visible in one figure.
    """
    output_dir = Path(output_dir)
    subset = _outer_only(predictions)
    if subset.empty or true_col not in subset.columns:
        _skip("residuals_by_condition", "columns missing or no outer rows"); return
    residual = (subset[pred_col].astype(float) - subset[true_col].astype(float)).abs()
    conditions = [
        ("temperature", "temperature (°C)", 5),
        ("DOD", "depth-of-discharge (%)", 4),
        ("discharge_C_rate", "discharge C-rate", 4),
    ]
    for column, xlabel, n_bins in conditions:
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce")
        mask = np.isfinite(values) & np.isfinite(residual)
        if mask.sum() < 5:
            _skip(f"residuals_vs_{column}", "not enough finite pairs"); continue
        try:
            bins = pd.qcut(values[mask], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        groups = [residual[mask][bins == cat].to_numpy()
                  for cat in bins.cat.categories]
        labels = [str(cat) for cat in bins.cat.categories]
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|residual| capacity (Ah)")
        ax.set_title(f"Residuals vs {column} (outer validation)")
        plt.xticks(rotation=25, ha="right")
        _save(fig, output_dir / f"residuals_vs_{column}")


def plot_rate_by_condition(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Predicted degradation rate binned by physical condition (T, DOD, C-rate)."""
    output_dir = Path(output_dir)
    subset = _outer_only(predictions)
    rate_col = ("predicted_degradation_rate_physical"
                if "predicted_degradation_rate_physical" in subset.columns
                else "predicted_degradation_rate_normalized")
    if subset.empty or rate_col not in subset.columns:
        _skip("rate_by_condition", "rate column missing"); return
    rate = pd.to_numeric(subset[rate_col], errors="coerce")
    for column, xlabel, n_bins in [
        ("temperature", "temperature (°C)", 5),
        ("DOD", "depth-of-discharge (%)", 4),
        ("discharge_C_rate", "discharge C-rate", 4),
    ]:
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce")
        mask = np.isfinite(values) & np.isfinite(rate)
        if mask.sum() < 5:
            continue
        try:
            bins = pd.qcut(values[mask], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        groups = [rate[mask][bins == cat].to_numpy() for cat in bins.cat.categories]
        labels = [str(cat) for cat in bins.cat.categories]
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rate_col)
        ax.set_title(f"Degradation rate vs {column} (outer validation)")
        plt.xticks(rotation=25, ha="right")
        _save(fig, output_dir / f"rate_vs_{column}")


def plot_integral_consistency(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Diagnostic: ∫r ds should approximately equal u_next - u_current for a PINN."""
    output_dir = Path(output_dir)
    subset = _outer_only(predictions)
    if subset.empty or "integral_consistency_error" not in subset.columns:
        _skip("integral_consistency", "columns missing"); return
    err = pd.to_numeric(subset["integral_consistency_error"], errors="coerce").dropna()
    if err.empty:
        _skip("integral_consistency", "no finite errors"); return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(err, bins=50, alpha=0.75)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("integral consistency error (u_next - ∫ r ds - u_current)")
    ax.set_ylabel("count")
    ax.set_title("Integral consistency diagnostic (outer validation)")
    _save(fig, output_dir / "integral_consistency_hist")


def plot_condition_trajectories(predictions: pd.DataFrame, output_dir: Path,
                                 *, target_col: str = "predicted_next_Q_Ah",
                                 true_col: str = "true_next_Q_Ah") -> None:
    """Per-condition mean trajectory vs stress, PINN vs observed."""
    output_dir = Path(output_dir)
    subset = _outer_only(predictions)
    if subset.empty or "condition_id" not in subset.columns \
            or "stress_target" not in subset.columns:
        _skip("condition_trajectories", "columns missing"); return
    conditions = sorted(subset["condition_id"].astype(str).unique())[:8]
    if not conditions:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for color, cond in zip(palette, conditions):
        sub = subset[subset["condition_id"].astype(str).eq(cond)].sort_values("stress_target")
        if sub.empty:
            continue
        ax.plot(sub["stress_target"], sub[true_col], color=color, linewidth=1.0,
                label=f"{cond} obs", alpha=0.9)
        ax.plot(sub["stress_target"], sub[target_col], color=color, linewidth=0.8,
                linestyle="--", alpha=0.7)
    ax.set_xlabel("stress at target time")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Per-condition observed (solid) vs predicted (dashed), outer validation")
    ax.legend(fontsize=6, loc="best", ncol=2)
    _save(fig, output_dir / "condition_trajectories")


def plot_accuracy_vs_runtime(complexity: pd.DataFrame, target_metrics: pd.DataFrame,
                              output_dir: Path,
                              *, metric: str = "MAE") -> None:
    """Scatter of (trainable parameters, mean MAE) — a rough accuracy/cost tradeoff."""
    output_dir = Path(output_dir)
    if complexity.empty or target_metrics.empty:
        _skip("accuracy_vs_runtime", "empty inputs"); return
    if "trainable_parameters" not in complexity.columns:
        _skip("accuracy_vs_runtime", "no trainable_parameters column"); return
    metrics = target_metrics
    if "target" in metrics.columns:
        metrics = metrics[metrics["target"].astype(str) == "next_rpt_Q_Ah"]
    if "aggregation" in metrics.columns:
        metrics = metrics[metrics["aggregation"].astype(str) == "cell_macro"]
    if "evaluation_role" in metrics.columns:
        metrics = metrics[metrics["evaluation_role"].astype(str) == DEFAULT_ROLE]
    if metrics.empty:
        _skip("accuracy_vs_runtime", "no rows after filters"); return
    per_arch_metric = metrics.groupby("architecture", as_index=False, dropna=False)[metric].mean()
    per_arch_params = complexity.groupby("architecture", as_index=False, dropna=False
                                          )["trainable_parameters"].mean()
    joined = per_arch_metric.merge(per_arch_params, on="architecture", how="inner")
    if joined.empty:
        _skip("accuracy_vs_runtime", "no architecture overlap"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(joined["trainable_parameters"], joined[metric], s=40)
    for _, r in joined.iterrows():
        ax.annotate(str(r["architecture"]),
                    (r["trainable_parameters"], r[metric]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(f"{metric} (outer validation, cell macro)")
    ax.set_title("Accuracy vs. model size")
    _save(fig, output_dir / f"accuracy_vs_size_{metric.lower()}")


def plot_combined_family_bar(combined_metrics: pd.DataFrame, output_dir: Path,
                              *, metric: str = "MAE") -> None:
    """Mean metric by architecture across all model families (classical + PINN)."""
    output_dir = Path(output_dir)
    if combined_metrics.empty or metric not in combined_metrics.columns:
        _skip("combined_family_bar", "columns missing"); return
    subset = combined_metrics
    if "target" in subset.columns:
        subset = subset[subset["target"].astype(str) == "next_rpt_Q_Ah"]
    if "aggregation" in subset.columns:
        subset = subset[subset["aggregation"].astype(str) == "cell_macro"]
    if "evaluation_role" in subset.columns:
        subset = subset[subset["evaluation_role"].astype(str).isin({DEFAULT_ROLE, ""})]
    if subset.empty or "architecture" not in subset.columns:
        _skip("combined_family_bar", "no rows after filters"); return
    summary = subset.groupby("architecture", as_index=False, dropna=False)[metric].mean().sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["architecture"].astype(str), summary[metric])
    ax.set_ylabel(f"mean {metric} (outer validation, cell macro)")
    ax.set_title("Model family comparison (ExtraTrees + NaPINN-Q)")
    plt.xticks(rotation=20, ha="right")
    _save(fig, output_dir / f"combined_family_{metric.lower()}")


def plot_training_curves(epoch_log: pd.DataFrame, output_dir: Path) -> None:
    """Combined training-loss + validation-MAE figure, one panel per fold/seed.

    Left axis: total training loss (log scale).
    Right axis: inner-validation MAE (linear scale).
    Placed in the main plots directory so it is immediately visible.
    """
    output_dir = Path(output_dir)
    if epoch_log.empty:
        _skip("training_curves", "epoch log is empty"); return
    required = {"epoch", "train_total_loss", "validation_MAE",
                "architecture", "preprocessing", "fold", "seed"}
    if not required.issubset(epoch_log.columns):
        missing = required - set(epoch_log.columns)
        _skip("training_curves", f"missing columns: {missing}"); return

    groups = list(epoch_log.groupby(
        ["architecture", "preprocessing", "fold", "seed"], dropna=False))
    n = len(groups)
    if n == 0:
        _skip("training_curves", "no groups"); return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for idx, ((arch, prep, fold, seed), grp) in enumerate(groups):
        ax = flat_axes[idx]
        grp = grp.sort_values("epoch")
        color_train = "tab:blue"
        color_val = "tab:orange"

        ax.set_xlabel("epoch")
        ax.set_ylabel("train total loss", color=color_train)
        ax.tick_params(axis="y", labelcolor=color_train)
        ax.set_yscale("log")
        ax.plot(grp["epoch"], grp["train_total_loss"],
                color=color_train, linewidth=1.0, label="train loss")

        ax2 = ax.twinx()
        ax2.set_ylabel("val MAE", color=color_val)
        ax2.tick_params(axis="y", labelcolor=color_val)
        ax2.plot(grp["epoch"], grp["validation_MAE"],
                 color=color_val, linewidth=1.0, linestyle="--", label="val MAE")

        ax.set_title(f"{arch}|{prep}|f{fold}|s{seed}", fontsize=8)

    for idx in range(n, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle("Training loss (blue, log) vs inner-validation MAE (orange, linear)", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "training_curves")


def render_all_pinn_plots(*, epoch_log: pd.DataFrame,
                          predictions: pd.DataFrame,
                          target_metrics: pd.DataFrame,
                          ablation_metrics: pd.DataFrame,
                          output_dir: Path,
                          complexity: pd.DataFrame | None = None,
                          combined_metrics: pd.DataFrame | None = None) -> None:
    """Render the pruned publication plot set into ``output_dir``.

    Removed plots (residual distribution, degradation-rate, seed stability,
    model complexity, accuracy-vs-runtime, rate-by-condition, condition
    trajectories) are intentionally not rendered. Training-history curves
    are routed to ``output_dir / "supplementary"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = output_dir / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)
    # Combined training-loss + validation-MAE curve in the main directory.
    plot_training_curves(epoch_log, output_dir)
    # Per-component training / validation / LR history → supplementary material.
    plot_epoch_curves(epoch_log, supp_dir)
    # Predicted vs actual capacity + SOH (same routine plots both when the
    # aggregator supplies the SOH columns).
    plot_predicted_vs_true(predictions, output_dir)
    # Observed vs predicted trajectories for unseen cells (capacity + SOH).
    plot_per_cell_trajectories(predictions, output_dir,
                                target_col="predicted_next_Q_Ah",
                                true_col="true_next_Q_Ah")
    if "predicted_next_SOH_pct" in predictions.columns and "true_next_SOH_pct" in predictions.columns:
        plot_per_cell_trajectories(
            predictions, output_dir / "soh",
            target_col="predicted_next_SOH_pct",
            true_col="true_next_SOH_pct",
        )
    # PDE-residual + integral-consistency figures.
    plot_pde_residual(predictions, output_dir)
    plot_pde_residual_vs_stress(predictions, output_dir)
    plot_integral_consistency(predictions, output_dir)
    # Error analysis across temperature, DOD, C-rate.
    plot_residuals_by_condition(predictions, output_dir)
    # Ablation results.
    plot_ablation_bar(ablation_metrics, output_dir)
    # Baseline family comparison.
    if combined_metrics is not None:
        plot_combined_family_bar(combined_metrics, output_dir)
