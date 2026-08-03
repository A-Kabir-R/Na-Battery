"""Manuscript figures for the studies that previously had no plotting at all.

The existing plotting covers training diagnostics and parity plots. None of the
paper's actual claims had a figure: the generalization study, the learning
curves, the coverage analysis and the robustness sweep all wrote CSVs that
nothing rendered.

Every figure here reads a **frozen results table**, never a model. That is what
makes `make figures` reproducible without a GPU, and it means a figure and the
table it came from cannot disagree.

Design rules applied throughout, because they are what the claims require:

* Error bars are cell-clustered. Treating anchors as independent would shrink
  every interval by roughly sqrt(anchors-per-cell).
* Persistence is drawn on every accuracy figure. On this dataset it wins outright
  in the noise-dominated conditions, so a figure without it overstates the result.
* Degenerate splits (a single held-out cell) are drawn hollow rather than
  dropped, so the reader sees the coverage of the study.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Vector first; the raster copy is a preview. Overridden by config in the script.
FORMATS: tuple[str, ...] = ("pdf", "png")
DEFAULT_DPI = 300

PERSISTENCE_COLOR = "#b03030"
NEUTRAL = "#4c72b0"


def save(fig, path: Path, *, formats: tuple[str, ...] = FORMATS,
         dpi: int = DEFAULT_DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _skip(name: str, reason: str) -> None:
    warnings.warn(f"[paper_figures] skipping {name}: {reason}", RuntimeWarning)


def _require(frame: pd.DataFrame | None, columns: tuple[str, ...],
             name: str) -> bool:
    if frame is None or frame.empty:
        _skip(name, "input table missing or empty")
        return False
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        _skip(name, f"missing columns {missing}")
        return False
    return True


# ---------------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------------

def fig_condition_matrix(anchors: pd.DataFrame, output: Path, **kw) -> None:
    """Cells and anchors per operating condition.

    The single most load-bearing dataset figure: it shows at a glance that six
    conditions carry one cell and one anchor, which is why those LOCO splits
    cannot be interpreted and why condition-macro metrics are unstable.
    """
    name = "condition_matrix"
    if not _require(anchors, ("condition", "cell", "T_degC", "DOD_pct"), name):
        return
    summary = (
        anchors.groupby(["DOD_pct", "T_degC"])
        .agg(cells=("cell", "nunique"), anchors=("condition", "size"))
        .reset_index()
    )
    dods = sorted(summary["DOD_pct"].unique())
    temps = sorted(summary["T_degC"].unique())
    grid = np.full((len(dods), len(temps)), np.nan)
    labels = np.empty_like(grid, dtype=object)
    for _, row in summary.iterrows():
        i, j = dods.index(row["DOD_pct"]), temps.index(row["T_degC"])
        grid[i, j] = row["anchors"]
        labels[i, j] = f"{int(row['anchors'])}\n({int(row['cells'])} cells)"

    fig, ax = plt.subplots(figsize=(1.6 * len(temps) + 2, 1.1 * len(dods) + 2))
    image = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(temps)), [f"{t} °C" for t in temps])
    ax.set_yticks(range(len(dods)), [f"DOD {d}%" for d in dods])
    for i in range(len(dods)):
        for j in range(len(temps)):
            if labels[i, j] is None:
                ax.text(j, i, "—", ha="center", va="center", color="0.6")
                continue
            # Single-cell conditions are the ones that cannot carry a claim.
            single = "(1 cells)" in labels[i, j]
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8,
                    color="crimson" if single else "black",
                    fontweight="bold" if single else "normal")
    ax.set_title("Anchors per operating condition\n"
                 "(red: a single cell — not interpretable alone)")
    fig.colorbar(image, ax=ax, label="anchors")
    save(fig, output / name, **kw)


def fig_signal_to_noise_regimes(anchors: pd.DataFrame, output: Path,
                                *, target: str = "delta_next_rpt_Q_Ah",
                                **kw) -> None:
    """Within-cell lag-1 autocorrelation of the target, per condition.

    The paper's first contribution in one figure. A value near -0.5 is the
    algebraic signature of a target that is pure RPT measurement noise: if
    ``y_k = c_k + e_k`` and the true increments are small, the differenced series
    is an MA(1) with rho = -0.5. Positive values mean real persistent
    degradation. Conditions left of the dashed line cannot be predicted by
    anything, which is why persistence wins there.
    """
    name = "signal_to_noise_regimes"
    if not _require(anchors, ("condition", "cell", target), name):
        return
    rows = []
    for (condition, cell), group in anchors.groupby(["condition", "cell"]):
        values = group[target].dropna().to_numpy(dtype=float)
        if values.size < 5:
            continue
        centered = values - values.mean()
        denominator = float((centered * centered).sum())
        if denominator <= 0:
            continue
        rows.append({
            "condition": condition,
            "lag1": float((centered[:-1] * centered[1:]).sum() / denominator),
        })
    if not rows:
        _skip(name, "no cell has >= 5 anchors")
        return
    frame = pd.DataFrame(rows)
    order = frame.groupby("condition")["lag1"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 2))
    positions = np.arange(len(order))
    ax.barh(positions, order.to_numpy(),
            color=[PERSISTENCE_COLOR if v < -0.2 else NEUTRAL for v in order])
    ax.set_yticks(positions, order.index)
    ax.axvline(-0.5, ls="--", c="0.3", lw=1)
    ax.axvline(0.0, c="0.6", lw=0.8)
    ax.text(-0.5, len(order) - 0.4, "  pure measurement noise",
            fontsize=8, color="0.3", va="top")
    ax.set_xlabel("within-cell lag-1 autocorrelation of Δcapacity")
    ax.set_title("Which conditions carry a predictable signal")
    save(fig, output / name, **kw)


# ---------------------------------------------------------------------------
# Generalization
# ---------------------------------------------------------------------------

def fig_generalization(metrics: pd.DataFrame, output: Path, *, study: str = "loco",
                       **kw) -> None:
    """Per-split error against the persistence floor.

    Drawn per split rather than as a mean, because the mean hides that
    persistence wins outright on the low-DOD conditions.
    """
    name = f"generalization_{study}"
    needed = ("study", "split_name", "model", "MAE", "persistence_MAE", "degenerate")
    if not _require(metrics, needed, name):
        return
    frame = metrics[metrics["study"].eq(study)]
    if frame.empty:
        _skip(name, f"no rows for study={study}")
        return

    models = [m for m in frame["model"].unique() if m != "ZeroChange"]
    splits = (
        frame.groupby("split_name")["persistence_MAE"].first().sort_values().index
    )
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(splits) + 3))
    positions = np.arange(len(splits))
    width = 0.8 / max(len(models), 1)
    for k, model in enumerate(models):
        subset = frame[frame["model"].eq(model)].set_index("split_name")
        values = [subset["MAE"].get(s, np.nan) for s in splits]
        degenerate = [bool(subset["degenerate"].get(s, False)) for s in splits]
        bars = ax.barh(positions + k * width - 0.4, values, height=width,
                       label=model)
        # Hollow bars mark splits with a single held-out cell: shown for
        # coverage, not to be read as an estimate.
        for bar, is_degenerate in zip(bars, degenerate):
            if is_degenerate:
                bar.set_alpha(0.35)
                bar.set_hatch("//")
    floor = [frame[frame["split_name"].eq(s)]["persistence_MAE"].iloc[0] for s in splits]
    ax.scatter(floor, positions, marker="|", s=260, color=PERSISTENCE_COLOR,
               zorder=5, label="persistence floor")
    ax.set_yticks(positions, splits, fontsize=8)
    ax.set_xlabel("MAE (Ah) on the held-out split")
    ax.set_title(f"{study.upper()}: error vs doing nothing\n"
                 "(hatched = single held-out cell)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, output / name, **kw)


# ---------------------------------------------------------------------------
# Data efficiency
# ---------------------------------------------------------------------------

def fig_low_data_curve(curve: pd.DataFrame, output: Path,
                       *, persistence_mae: float | None = None, **kw) -> None:
    """Error against the number of training cells, with an IQR band."""
    name = "low_data_learning_curve"
    if not _require(curve, ("model", "median_train_cells", "median", "q25", "q75"),
                    name):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, group in curve.groupby("model"):
        group = group.sort_values("median_train_cells")
        ax.plot(group["median_train_cells"], group["median"], marker="o", label=model)
        ax.fill_between(group["median_train_cells"], group["q25"], group["q75"],
                        alpha=0.15)
    if persistence_mae is not None:
        ax.axhline(persistence_mae, ls="--", color=PERSISTENCE_COLOR,
                   label="persistence")
    ax.set_xlabel("training cells")
    ax.set_ylabel("cell-macro MAE (Ah)")
    ax.set_title("Data efficiency: error vs aging-test burden\n"
                 "(band = inter-quartile range over repeated cell draws)")
    ax.legend(fontsize=8)
    save(fig, output / name, **kw)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def fig_calibration(coverage: pd.DataFrame, output: Path, *, alpha: float = 0.1,
                    **kw) -> None:
    """Empirical vs nominal coverage per condition.

    Marginal coverage can sit on nominal while a whole condition is at zero.
    That is the failure this figure exists to expose, and it is what happens
    here on the single-anchor extreme conditions.
    """
    name = "calibration_by_condition"
    if not _require(coverage, ("model", "condition", "PICP", "n"), name):
        return
    # One shared condition order for every model. Sorting each model's rows
    # independently puts a different condition on row i per model, so points
    # that look comparable are not -- and with no y tick labels there is nothing
    # to catch it. Order by worst mean coverage so the failures sit together.
    order = (
        coverage.groupby("condition")["PICP"].mean().sort_values().index.tolist()
    )
    position_of = {condition: i for i, condition in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 2.4))
    for model, group in coverage.groupby("model"):
        positions = group["condition"].map(position_of).to_numpy(dtype=float)
        ax.scatter(group["PICP"], positions,
                   s=np.clip(group["n"].to_numpy(dtype=float) * 4, 20, 260),
                   alpha=0.75, label=model)
    ax.axvline(1.0 - alpha, ls="--", color="0.3",
               label=f"nominal {100 * (1 - alpha):.0f}%")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("empirical coverage (PICP)")
    ax.set_ylabel("condition")
    ax.set_title("Coverage per condition\n(marker area ∝ anchors)")
    ax.legend(fontsize=8)
    save(fig, output / name, **kw)


def fig_prediction_intervals(predictions: pd.DataFrame, output: Path,
                             *, max_cells: int = 12, **kw) -> None:
    """Observed vs predicted with conformal intervals, per cell."""
    name = "prediction_intervals"
    if not _require(predictions, ("cell", "y_true", "y_pred", "lower", "upper"),
                    name):
        return
    cells = list(predictions["cell"].unique())[:max_cells]
    subset = predictions[predictions["cell"].isin(cells)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(subset))
    ax.vlines(positions, subset["lower"], subset["upper"], color="0.7", lw=2)
    ax.scatter(positions, subset["y_pred"], s=14, label="predicted", zorder=4)
    covered = (subset["y_true"] >= subset["lower"]) & (subset["y_true"] <= subset["upper"])
    ax.scatter(positions[covered], subset["y_true"][covered], s=14, marker="x",
               color="black", label="observed (covered)", zorder=5)
    ax.scatter(positions[~covered], subset["y_true"][~covered], s=40, marker="x",
               color=PERSISTENCE_COLOR, label="observed (missed)", zorder=6)
    ax.set_xlabel("anchor")
    ax.set_ylabel("Δcapacity (Ah)")
    ax.set_title("Conformal prediction intervals")
    ax.legend(fontsize=8)
    save(fig, output / name, **kw)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def fig_robustness(summary: pd.DataFrame, output: Path, **kw) -> None:
    """Error inflation per perturbation, with interval widening alongside.

    The pass criterion is not "error stayed low" but "the intervals widened".
    A model that loses a diagnostic and stays equally confident is the failure
    mode worth reporting.
    """
    name = "robustness"
    if not _require(summary, ("model", "perturbation", "mean_MAE_ratio"), name):
        return
    pivot = summary.pivot_table(index="perturbation", columns="model",
                                values="mean_MAE_ratio", aggfunc="mean")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    has_width = "mean_width_ratio" in summary.columns and \
        summary["mean_width_ratio"].notna().any()
    fig, axes = plt.subplots(
        1, 2 if has_width else 1, figsize=(13 if has_width else 7.5,
                                           0.36 * len(pivot) + 2.5),
        squeeze=False,
    )
    ax = axes[0][0]
    pivot.plot.barh(ax=ax, width=0.8)
    ax.axvline(1.0, ls="--", color="0.3")
    ax.set_xlabel("MAE ratio vs clean inputs")
    ax.set_title("Error inflation under perturbation")
    ax.legend(fontsize=8)

    if has_width:
        width_pivot = summary.pivot_table(index="perturbation", columns="model",
                                          values="mean_width_ratio", aggfunc="mean")
        width_pivot = width_pivot.reindex(pivot.index)
        ax2 = axes[0][1]
        width_pivot.plot.barh(ax=ax2, width=0.8)
        ax2.axvline(1.0, ls="--", color="0.3")
        ax2.set_xlabel("interval-width ratio")
        ax2.set_title("Did the model admit the loss?\n(> 1 is the desired behaviour)")
        ax2.legend(fontsize=8)
    save(fig, output / name, **kw)


# ---------------------------------------------------------------------------
# Physics interpretation
# ---------------------------------------------------------------------------

def fig_rate_decomposition(components: pd.DataFrame, output: Path, **kw) -> None:
    """Stacked degradation-rate components over life.

    The interpretability deliverable: if the neural residual carries most of the
    rate, the model is a black box wearing a physics label, and this figure says
    so immediately.
    """
    name = "rate_decomposition"
    if not _require(components, ("stress", "cycling", "residual"), name):
        return
    available = [c for c in ("cycling", "cold", "calendar", "diagnostic", "residual")
                 if c in components.columns]
    frame = components.sort_values("stress")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(frame["stress"], *[frame[c] for c in available], labels=available,
                 alpha=0.85)
    ax.set_xlabel("stress coordinate (EFC)")
    ax.set_ylabel("degradation rate (per EFC)")
    ax.set_title("Rate decomposition")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, output / name, **kw)


def fig_response_surface(surface: pd.DataFrame, output: Path, *, x: str, y: str,
                         value: str = "rate", **kw) -> None:
    """Model-evaluated rate over a physical grid, with data support marked.

    Regions outside the convex support of the training conditions are hatched:
    a response surface read outside its support is extrapolation, and the figure
    must say which is which.
    """
    name = f"response_surface_{x}_{y}"
    if not _require(surface, (x, y, value), name):
        return
    pivot = surface.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.to_numpy(),
                         shading="nearest", cmap="viridis")
    if "supported" in surface.columns:
        unsupported = surface[~surface["supported"].astype(bool)]
        ax.scatter(unsupported[x], unsupported[y], marker="x", s=12,
                   color="white", alpha=0.6, label="outside data support")
        ax.legend(fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Predicted degradation rate over ({x}, {y})")
    fig.colorbar(mesh, ax=ax, label=value)
    save(fig, output / name, **kw)


def fig_paired_comparison(per_cell_error: pd.DataFrame, output: Path,
                          *, reference: str = "PreviousRPT",
                          n_boot: int = 4000, seed: int = 0, **kw) -> None:
    """Paired cell-clustered bootstrap of each model against a reference.

    The only comparison on this dataset that survives clustering is
    model-versus-persistence; between the top model families the intervals cross
    zero. Drawing the interval rather than a bar is what makes that visible.
    """
    name = "paired_comparison"
    if not _require(per_cell_error, ("model", "cell", "abs_error"), name):
        return
    pivot = per_cell_error.pivot_table(index="cell", columns="model",
                                       values="abs_error", aggfunc="mean")
    if reference not in pivot.columns:
        _skip(name, f"reference model {reference} absent")
        return
    rng = np.random.default_rng(seed)
    cells = pivot.index.to_numpy()
    draws = [rng.choice(len(cells), len(cells), replace=True) for _ in range(n_boot)]

    rows = []
    for model in pivot.columns:
        if model == reference:
            continue
        paired = (pivot[reference] - pivot[model]).dropna()
        if paired.empty:
            continue
        values = paired.to_numpy()
        boot = np.array([values[idx[idx < len(values)]].mean() for idx in draws])
        rows.append({
            "model": model,
            "improvement": float(values.mean()),
            "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5)),
        })
    if not rows:
        _skip(name, "no comparable models")
        return
    frame = pd.DataFrame(rows).sort_values("improvement")

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(frame) + 2.2))
    positions = np.arange(len(frame))
    ax.errorbar(frame["improvement"], positions,
                xerr=[frame["improvement"] - frame["low"],
                      frame["high"] - frame["improvement"]],
                fmt="o", capsize=4, color=NEUTRAL)
    ax.axvline(0.0, ls="--", color=PERSISTENCE_COLOR)
    ax.set_yticks(positions, frame["model"])
    ax.set_xlabel(f"MAE improvement over {reference} (Ah)")
    ax.set_title("Paired cell-clustered bootstrap\n"
                 "(intervals crossing zero are not resolvable)")
    save(fig, output / name, **kw)
