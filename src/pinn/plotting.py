"""Publication-quality plots for NaPINN-Q.

Every plot is written as PNG (300 dpi) plus PDF. Missing data raises a warning
but does not crash the pipeline — combined reports must remain resilient to
partial runs.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


FORMATS = ("png", "pdf")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _skip(name: str, reason: str) -> None:
    warnings.warn(f"[pinn.plot] skipping {name}: {reason}", RuntimeWarning)


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
        ("GPU_memory_allocated_MB", "GPU memory allocated (MB)", False),
    ]:
        _epoch_curve(epoch_log, column, title, output_dir / f"epoch_{column}", log_y=log_y)


def plot_predicted_vs_true(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    for target, true_col, pred_col in [
        ("capacity", "true_next_Q_Ah", "predicted_next_Q_Ah"),
        ("soh", "true_next_SOH_pct", "predicted_next_SOH_pct"),
    ]:
        if predictions.empty or true_col not in predictions.columns:
            _skip(f"pred_vs_true_{target}", "columns missing"); continue
        fig, ax = plt.subplots(figsize=(5, 5))
        mask = predictions["evaluation_role"].astype(str).eq("outer_validation")
        subset = predictions[mask]
        ax.scatter(subset[true_col], subset[pred_col], s=6, alpha=0.5)
        lo = float(min(subset[true_col].min(), subset[pred_col].min()))
        hi = float(max(subset[true_col].max(), subset[pred_col].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(true_col)
        ax.set_ylabel(pred_col)
        ax.set_title(f"Predicted vs observed ({target})")
        _save(fig, output_dir / f"predicted_vs_true_{target}")


def plot_residual_distribution(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    for target, true_col, pred_col in [
        ("capacity", "true_next_Q_Ah", "predicted_next_Q_Ah"),
        ("soh", "true_next_SOH_pct", "predicted_next_SOH_pct"),
    ]:
        if predictions.empty or true_col not in predictions.columns:
            _skip(f"residual_{target}", "columns missing"); continue
        residual = (predictions[pred_col] - predictions[true_col]).astype(float).dropna()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(residual, bins=40, alpha=0.75)
        ax.set_xlabel(f"residual ({target})")
        ax.set_ylabel("count")
        ax.set_title(f"Residual distribution ({target})")
        _save(fig, output_dir / f"residual_hist_{target}")


def plot_per_cell_trajectories(predictions: pd.DataFrame, output_dir: Path,
                                target_col: str = "predicted_next_Q_Ah",
                                true_col: str = "true_next_Q_Ah") -> None:
    output_dir = Path(output_dir)
    if predictions.empty or true_col not in predictions.columns:
        _skip("per_cell_trajectories", "columns missing"); return
    cells = sorted(predictions["cell_id"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for cell in cells[:30]:
        sub = predictions[predictions["cell_id"].astype(str).eq(cell)].sort_values("stress_current")
        ax.plot(sub["stress_current"], sub[true_col], color="black", alpha=0.4, linewidth=0.6)
        ax.plot(sub["stress_current"], sub[target_col], color="tab:red", alpha=0.4, linewidth=0.6)
    ax.set_xlabel("stress (physical units)")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Per-cell observed (black) vs predicted (red) capacity")
    _save(fig, output_dir / "per_cell_trajectories")


def plot_degradation_rate(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if predictions.empty or "predicted_degradation_rate" not in predictions.columns:
        _skip("degradation_rate", "columns missing"); return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(predictions["predicted_degradation_rate"].astype(float).dropna(),
            bins=50, alpha=0.75)
    ax.set_xlabel("predicted effective degradation rate")
    ax.set_ylabel("count")
    ax.set_title("Predicted degradation-rate distribution")
    _save(fig, output_dir / "degradation_rate_distribution")


def plot_pde_residual(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if predictions.empty or "pde_residual" not in predictions.columns:
        _skip("pde_residual", "columns missing"); return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(predictions["pde_residual"].astype(float).dropna(), bins=50, alpha=0.75)
    ax.set_xlabel("PDE residual (du/ds + r)")
    ax.set_ylabel("count")
    ax.set_title("Governing-equation residual distribution")
    _save(fig, output_dir / "pde_residual_distribution")


def plot_ablation_bar(ablation_metrics: pd.DataFrame, output_dir: Path,
                      metric: str = "MAE") -> None:
    output_dir = Path(output_dir)
    if ablation_metrics.empty or metric not in ablation_metrics.columns:
        _skip("ablation_bar", "columns missing"); return
    subset = ablation_metrics[ablation_metrics["target"] == "next_rpt_Q_Ah"]
    if subset.empty:
        _skip("ablation_bar", "no rows"); return
    fig, ax = plt.subplots(figsize=(7, 4))
    order = subset.sort_values(metric)["ablation"].tolist()
    values = subset.set_index("ablation").reindex(order)[metric]
    ax.bar(order, values)
    ax.set_ylabel(metric)
    ax.set_title(f"Physics ablation ({metric})")
    plt.xticks(rotation=30, ha="right")
    _save(fig, output_dir / f"ablation_{metric.lower()}")


def plot_seed_stability(target_metrics: pd.DataFrame, output_dir: Path,
                         metric: str = "MAE") -> None:
    output_dir = Path(output_dir)
    if target_metrics.empty or "seed" not in target_metrics.columns:
        _skip("seed_stability", "column missing"); return
    subset = target_metrics[target_metrics["target"] == "next_rpt_Q_Ah"]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [g[metric].dropna().to_numpy() for _, g in subset.groupby("architecture")]
    labels = [name for name, _ in subset.groupby("architecture")]
    if not groups:
        _skip("seed_stability", "no groups"); return
    ax.boxplot(groups, labels=labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Seed / fold stability ({metric})")
    _save(fig, output_dir / f"seed_stability_{metric.lower()}")


def render_all_pinn_plots(*, epoch_log: pd.DataFrame,
                          predictions: pd.DataFrame,
                          target_metrics: pd.DataFrame,
                          ablation_metrics: pd.DataFrame,
                          output_dir: Path) -> None:
    """Render every PINN plot into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_epoch_curves(epoch_log, output_dir)
    plot_predicted_vs_true(predictions, output_dir)
    plot_residual_distribution(predictions, output_dir)
    plot_per_cell_trajectories(predictions, output_dir)
    plot_degradation_rate(predictions, output_dir)
    plot_pde_residual(predictions, output_dir)
    plot_ablation_bar(ablation_metrics, output_dir)
    plot_seed_stability(target_metrics, output_dir)
