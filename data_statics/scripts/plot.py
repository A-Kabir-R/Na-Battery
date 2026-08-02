"""
Generate every plot for the Standard-cycling dataset from the tables written
by analyze.py.

  plots/
    inventory/          bar charts of files-per-condition, cells, CU vs CYC
    signals/            per-condition boxplots of V, I, T, Ah + histograms
    sample_waveforms/   raw V/I/T traces from a representative CYC file per condition
    cycles/             per-condition Q_discharge, CE, EE, T_max vs cycle
    capacity_fade/      SOH trajectories (per-condition panels + overlay)
    heatmaps/           T x C-rate degradation heatmap, EOL heatmap
    comparisons/        DOE bar charts (impact of T, C-rate, DOD)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = Path(__file__).resolve().parents[2]
TABLES = OUT / "tables"
PLOTS = OUT / "plots"
for sub in ("inventory", "signals", "sample_waveforms", "cycles",
            "capacity_fade", "heatmaps", "comparisons"):
    (PLOTS / sub).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
    "figure.autolayout": True,
})

# ---------------------------------------------------------------- helpers

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT)}")


def _cond_order(df: pd.DataFrame) -> list[str]:
    """Sort conditions by (DOD, C_ch, T) for consistent x-axis."""
    from analyze import parse_condition
    conds = sorted(df["condition"].dropna().unique().tolist())
    decoded = [(c, parse_condition(c)) for c in conds]
    decoded.sort(key=lambda x: (x[1]["DOD_pct"], x[1]["C_ch"], x[1]["T_degC"]))
    return [c for c, _ in decoded]


def _color_map(items: list[str]) -> dict[str, tuple]:
    cmap = cm.get_cmap("tab20", max(len(items), 3))
    return {it: cmap(i) for i, it in enumerate(items)}


# ---------------------------------------------------------------- inventory

def plot_inventory(inv: pd.DataFrame) -> None:
    order = _cond_order(inv)

    # files per condition
    counts = inv.groupby("condition").size().reindex(order)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(counts)), counts.values, color="steelblue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("number of .ird files")
    ax.set_title("File count per test condition")
    _save(fig, PLOTS / "inventory" / "files_per_condition.png")

    # CU vs CYC
    ct = (inv.groupby(["condition", "test_type"]).size()
             .unstack(fill_value=0).reindex(order))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(ct))
    w = 0.4
    if "CU"  in ct.columns: ax.bar(x - w/2, ct["CU"],  w, label="CU (RPT)", color="steelblue")
    if "CYC" in ct.columns: ax.bar(x + w/2, ct["CYC"], w, label="CYC (aging)", color="salmon")
    ax.set_xticks(x); ax.set_xticklabels(ct.index, rotation=45, ha="right")
    ax.set_ylabel("file count"); ax.set_title("CU vs CYC files per condition"); ax.legend()
    _save(fig, PLOTS / "inventory" / "cu_vs_cyc_per_condition.png")

    # cells per condition
    cells = inv.groupby("condition")["cell"].nunique().reindex(order)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(cells)), cells.values, color="seagreen")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells.index, rotation=45, ha="right")
    ax.set_ylabel("unique cells"); ax.set_title("Cells per condition")
    _save(fig, PLOTS / "inventory" / "cells_per_condition.png")

    # visits per cell (grouped by condition)
    vpc = inv.groupby(["condition", "cell"]).size().reset_index(name="visits")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (cond, g) in enumerate(vpc.groupby("condition")):
        ax.scatter([i] * len(g), g["visits"], alpha=0.6, s=45)
    ax.set_xticks(range(vpc["condition"].nunique()))
    ax.set_xticklabels(sorted(vpc["condition"].unique()), rotation=45, ha="right")
    ax.set_ylabel("visits per cell"); ax.set_title("Visits per cell, by condition")
    _save(fig, PLOTS / "inventory" / "visits_per_cell.png")

    # file size distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in ("CU", "CYC"):
        s = inv.loc[inv["test_type"] == t, "size_MB"]
        if len(s):
            ax.hist(s, bins=40, alpha=0.55, label=f"{t} (n={len(s)})")
    ax.set_xlabel("file size (MB)"); ax.set_ylabel("count")
    ax.set_title("File-size distribution"); ax.legend()
    _save(fig, PLOTS / "inventory" / "file_size_hist.png")


# ---------------------------------------------------------------- signals

def plot_signals(stats: pd.DataFrame) -> None:
    stats = stats[stats["parse_status"] == "ok"].copy()
    order = _cond_order(stats)

    def box_by_cond(col: str, ylabel: str, title: str, fname: str, ttype: Optional[str] = None) -> None:
        d = stats if ttype is None else stats[stats["test_type"] == ttype]
        data = [d.loc[d["condition"] == c, col].dropna().values for c in order]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.boxplot(data, labels=order, showmeans=True)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title)
        _save(fig, PLOTS / "signals" / fname)

    box_by_cond("V_mean", "V_mean (V)",           "Mean voltage per file, by condition",
                "V_mean_by_condition.png")
    box_by_cond("V_max",  "V_max (V)",            "Max voltage per file, by condition",
                "V_max_by_condition.png")
    box_by_cond("V_min",  "V_min (V)",            "Min voltage per file, by condition",
                "V_min_by_condition.png")
    box_by_cond("T_mean", "T_mean (deg C)",      "Mean temperature per file, by condition (CYC only)",
                "T_mean_by_condition_CYC.png", ttype="CYC")
    box_by_cond("T_max",  "T_max (deg C)",       "Peak temperature per file, by condition (CYC only)",
                "T_max_by_condition_CYC.png", ttype="CYC")
    box_by_cond("I_max",  "I_max (A)",           "Peak charge current per file, by condition",
                "I_max_by_condition.png")
    box_by_cond("I_min",  "I_min (A)",           "Peak discharge current per file, by condition",
                "I_min_by_condition.png")
    box_by_cond("Ah_span","Ah span (Ah)",        "Ah span per file, by condition",
                "Ah_span_by_condition.png")
    box_by_cond("dt_median_s", "dt median (s)",  "Median sample period per file",
                "dt_median_by_condition.png")

    # Global histograms (all files pooled)
    for col, xlabel, fname in [
        ("V_mean",  "V_mean per file (V)",     "hist_V_mean.png"),
        ("V_max",   "V_max per file (V)",      "hist_V_max.png"),
        ("T_mean",  "T_mean per file (deg C)", "hist_T_mean.png"),
        ("T_max",   "T_max per file (deg C)",  "hist_T_max.png"),
        ("dt_median_s", "dt median (s)",       "hist_dt_median.png"),
        ("rows",    "rows per file",           "hist_rows.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(stats[col].dropna(), bins=40, color="steelblue", edgecolor="k")
        ax.set_xlabel(xlabel); ax.set_ylabel("files"); ax.set_title(f"Histogram: {xlabel}")
        _save(fig, PLOTS / "signals" / fname)


# ---------------------------------------------------------------- sample waveforms

def plot_sample_waveforms(inv: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Pick the first parse-OK CYC file for each condition, plot V/I/T vs time."""
    from analyze import read_ird       # reuse parser

    order = _cond_order(inv)
    stats_ok = stats[(stats["parse_status"] == "ok") & (stats["test_type"] == "CYC")]

    for cond in order:
        row = stats_ok[stats_ok["condition"] == cond].sort_values("visit").head(1)
        if not len(row):
            continue
        p = Path(row.iloc[0]["path"])
        try:
            df = read_ird(p)
        except Exception as e:
            print(f"  [sample] skip {cond}: {e}")
            continue
        # First ~6 hours of data to keep plot legible
        t0 = df["Time"].iloc[0]
        end = t0 + 6 * 3600
        df = df[df["Time"] <= end]
        if len(df) < 100:
            df = df.iloc[:200_000]
        t_h = (df["Time"] - df["Time"].iloc[0]) / 3600.0

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_h, df["Voltage"], color="tab:blue",   lw=0.8); axes[0].set_ylabel("V (V)")
        axes[1].plot(t_h, df["Current"], color="tab:orange", lw=0.8); axes[1].set_ylabel("I (A)")
        axes[2].plot(t_h, df["Temperature"], color="tab:red",lw=0.8); axes[2].set_ylabel("T (deg C)")
        axes[2].set_xlabel("Time (h)")
        fig.suptitle(f"Sample CYC waveform — {cond}  |  cell {row.iloc[0]['cell']}  V{int(row.iloc[0]['visit']):02d}")
        _save(fig, PLOTS / "sample_waveforms" / f"waveform_{cond}.png")

        # V vs Ah for the same window
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Ah_counter"], df["Voltage"], lw=0.4, color="steelblue")
        ax.set_xlabel("Ah_counter (Ah)"); ax.set_ylabel("Voltage (V)")
        ax.set_title(f"V vs Ah — {cond}")
        _save(fig, PLOTS / "sample_waveforms" / f"V_vs_Ah_{cond}.png")


# ---------------------------------------------------------------- cycles

def plot_cycles(cyc: pd.DataFrame) -> None:
    if not len(cyc):
        print("  [cycles] no cycle rows -> skipping")
        return
    order = _cond_order(cyc)

    for cond in order:
        g = cyc[cyc["condition"] == cond]
        if not len(g):
            continue
        cells = sorted(g["cell"].unique())
        col = _color_map(cells)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for cell in cells:
            gc = g[g["cell"] == cell].sort_values("global_cycle")
            axes[0, 0].plot(gc["global_cycle"], gc["Q_discharge_Ah"], marker=".", ms=2, lw=0.6, color=col[cell], label=cell)
            axes[0, 1].plot(gc["global_cycle"], gc["coulombic_efficiency"] * 100, marker=".", ms=2, lw=0.6, color=col[cell])
            axes[1, 0].plot(gc["global_cycle"], gc["energy_efficiency"]    * 100, marker=".", ms=2, lw=0.6, color=col[cell])
            axes[1, 1].plot(gc["global_cycle"], gc["T_max"],                  marker=".", ms=2, lw=0.6, color=col[cell])
        axes[0, 0].set_ylabel("Q_discharge (Ah)");  axes[0, 0].set_xlabel("cycle")
        axes[0, 1].set_ylabel("CE (%)");             axes[0, 1].set_xlabel("cycle")
        axes[1, 0].set_ylabel("Energy eff. (%)");    axes[1, 0].set_xlabel("cycle")
        axes[1, 1].set_ylabel("T_max (deg C)");      axes[1, 1].set_xlabel("cycle")
        axes[0, 0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Per-cycle metrics — {cond}")
        _save(fig, PLOTS / "cycles" / f"cycles_{cond}.png")


# ---------------------------------------------------------------- capacity fade

def plot_capacity_fade(cyc: pd.DataFrame, deg: pd.DataFrame) -> None:
    if not len(cyc):
        return
    order = _cond_order(cyc)

    # Per-condition SOH vs cycle (panels)
    n = len(order); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.atleast_2d(axes)
    for i, cond in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        g = cyc[cyc["condition"] == cond]
        for cell, gc in g.groupby("cell"):
            gc = gc.sort_values("global_cycle")
            ax.plot(gc["global_cycle"], gc["SOH_pct"], lw=0.8, label=cell)
        ax.axhline(80, color="red", ls="--", lw=0.6)
        ax.set_title(cond, fontsize=8)
        ax.set_xlabel("cycle"); ax.set_ylabel("SOH (%)")
        ax.legend(fontsize=6)
    # blank unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("SOH (%) vs cycle — per condition")
    _save(fig, PLOTS / "capacity_fade" / "SOH_panels_by_condition.png")

    # Overlay: median SOH vs cycle per condition
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.get_cmap("tab20", len(order))
    for i, cond in enumerate(order):
        g = cyc[cyc["condition"] == cond]
        m = g.groupby("global_cycle")["SOH_pct"].median()
        ax.plot(m.index, m.values, lw=1.2, label=cond, color=cmap(i))
    ax.axhline(80, color="red", ls="--", lw=0.6, label="80% SOH")
    ax.set_xlabel("cycle"); ax.set_ylabel("median SOH (%)")
    ax.set_title("Median SOH trajectory — all conditions overlay")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, PLOTS / "capacity_fade" / "SOH_overlay_all_conditions.png")

    # SOH vs cumulative Ah throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cond in enumerate(order):
        g = cyc[cyc["condition"] == cond]
        m = (g.sort_values("cumulative_Ah_throughput")
               .groupby(pd.cut(g["cumulative_Ah_throughput"], bins=100))["SOH_pct"]
               .median())
        # use midpoints
        x = [iv.mid for iv in m.index if pd.notna(iv)]
        y = m.values
        if len(x) == len(y):
            ax.plot(x, y, lw=1.2, label=cond, color=cmap(i))
    ax.axhline(80, color="red", ls="--", lw=0.6)
    ax.set_xlabel("cumulative Ah throughput (Ah)"); ax.set_ylabel("median SOH (%)")
    ax.set_title("SOH vs cumulative Ah throughput")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, PLOTS / "capacity_fade" / "SOH_vs_throughput.png")

    # EOL bar chart
    if len(deg):
        eol = (deg.dropna(subset=["eol80_cycle"])
                  .groupby("condition")["eol80_cycle"].mean()
                  .reindex([c for c in order if c in deg["condition"].unique()])
                  .dropna())
        if len(eol):
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.bar(range(len(eol)), eol.values, color="tab:red")
            ax.set_xticks(range(len(eol)))
            ax.set_xticklabels(eol.index, rotation=45, ha="right")
            ax.set_ylabel("cycles to 80% SOH")
            ax.set_title("End-of-life (80% SOH) cycle count — condition mean")
            _save(fig, PLOTS / "capacity_fade" / "EOL_cycles_by_condition.png")

        # Fade slope bar
        slope = (deg.groupby("condition")["fade_slope_pct_per_cycle"].mean()
                    .reindex(order).dropna())
        fig, ax = plt.subplots(figsize=(11, 4.5))
        colors = ["tab:red" if v < 0 else "tab:green" for v in slope.values]
        ax.bar(range(len(slope)), slope.values, color=colors)
        ax.set_xticks(range(len(slope)))
        ax.set_xticklabels(slope.index, rotation=45, ha="right")
        ax.set_ylabel("SOH slope (%/cycle)")
        ax.set_title("Linear degradation slope — condition mean (more negative = faster fade)")
        _save(fig, PLOTS / "capacity_fade" / "fade_slope_by_condition.png")


# ---------------------------------------------------------------- heatmaps

def plot_heatmaps(deg: pd.DataFrame) -> None:
    if not len(deg):
        return

    def _grid(value_col: str, agg: str, title: str, fname: str, fmt: str = "{:.2f}") -> None:
        for dod in sorted(deg["DOD_pct"].dropna().unique()):
            sub = deg[deg["DOD_pct"] == dod]
            piv = sub.pivot_table(index="T_degC", columns="C_ch", values=value_col, aggfunc=agg)
            if piv.empty:
                continue
            fig, ax = plt.subplots(figsize=(1.4 * len(piv.columns) + 3, 0.8 * len(piv.index) + 2))
            im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c}C" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{t}°C" for t in piv.index])
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if pd.notna(v):
                        ax.text(j, i, fmt.format(v), ha="center", va="center",
                                color="white" if v < np.nanmedian(piv.values) else "black", fontsize=9)
            ax.set_xlabel("C-rate"); ax.set_ylabel("temperature")
            ax.set_title(f"{title}\nDOD={int(dod)}%")
            fig.colorbar(im, ax=ax, shrink=0.75)
            _save(fig, PLOTS / "heatmaps" / f"{fname}_DOD{int(dod)}.png")

    _grid("SOH_final_pct",           "mean", "Final SOH (%)",           "heatmap_SOH_final",  fmt="{:.1f}")
    _grid("fade_slope_pct_per_cycle","mean", "Fade slope (%/cycle)",    "heatmap_fade_slope", fmt="{:.3f}")
    _grid("eol80_cycle",             "mean", "Cycles to 80% SOH",       "heatmap_eol_cycle",  fmt="{:.0f}")
    _grid("Q_initial_Ah",            "mean", "Initial capacity (Ah)",   "heatmap_Q_initial",  fmt="{:.2f}")


# ---------------------------------------------------------------- comparisons (DOE)

def plot_comparisons(deg: pd.DataFrame) -> None:
    if not len(deg):
        return

    def _bar(factor: str, ylabel: str, value: str, agg: str, title: str, fname: str) -> None:
        g = deg.groupby(factor)[value].agg([agg, "std", "count"]).reset_index()
        g = g.sort_values(factor)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(g)), g[agg], yerr=g["std"].fillna(0), capsize=4, color="tab:blue")
        ax.set_xticks(range(len(g))); ax.set_xticklabels(g[factor].astype(str))
        ax.set_xlabel(factor); ax.set_ylabel(ylabel); ax.set_title(title)
        _save(fig, PLOTS / "comparisons" / fname)

    for factor, label in [("T_degC", "Temperature (deg C)"),
                          ("C_ch",   "Charge C-rate"),
                          ("DOD_pct","DOD (%)")]:
        _bar(factor, "Fade slope (%/cycle)",  "fade_slope_pct_per_cycle", "mean",
             f"Fade slope vs {label}",  f"fade_vs_{factor}.png")
        _bar(factor, "Cycles to 80% SOH",     "eol80_cycle",              "mean",
             f"EOL cycles vs {label}",  f"eol_vs_{factor}.png")
        _bar(factor, "Final SOH (%)",         "SOH_final_pct",            "mean",
             f"Final SOH vs {label}",   f"soh_vs_{factor}.png")

    # scatter: fade slope vs T colored by DOD, size by C-rate
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 40 + 60 * deg["C_ch"].fillna(1)
    sc = ax.scatter(deg["T_degC"], deg["fade_slope_pct_per_cycle"],
                    c=deg["DOD_pct"], s=sizes, cmap="plasma",
                    edgecolor="k", linewidths=0.4, alpha=0.85)
    ax.set_xlabel("Temperature (deg C)"); ax.set_ylabel("Fade slope (%/cycle)")
    ax.set_title("Degradation rate vs temperature\n(color = DOD %, size ∝ C-rate)")
    fig.colorbar(sc, ax=ax, label="DOD (%)")
    _save(fig, PLOTS / "comparisons" / "scatter_fade_vs_T.png")


# ---------------------------------------------------------------- main

def main() -> None:
    inv   = pd.read_csv(TABLES / "inventory.csv")
    stats = pd.read_csv(TABLES / "file_statistics.csv")
    print(f"[plot] inventory: {len(inv)} rows")
    print(f"[plot] stats:     {len(stats)} rows")

    cyc = pd.read_csv(TABLES / "cycle_metrics.csv") if (TABLES / "cycle_metrics.csv").exists() else pd.DataFrame()
    deg = pd.read_csv(TABLES / "degradation_rates.csv") if (TABLES / "degradation_rates.csv").exists() else pd.DataFrame()
    print(f"[plot] cycles:    {len(cyc)} rows")
    print(f"[plot] degradation: {len(deg)} rows")

    print("[plot] inventory ..."); plot_inventory(inv)
    print("[plot] signals ...");   plot_signals(stats)
    print("[plot] sample waveforms ..."); plot_sample_waveforms(inv, stats)
    print("[plot] per-cycle ...");        plot_cycles(cyc)
    print("[plot] capacity fade ...");    plot_capacity_fade(cyc, deg)
    print("[plot] heatmaps ...");         plot_heatmaps(deg)
    print("[plot] comparisons ...");      plot_comparisons(deg)
    print("[plot] done")


if __name__ == "__main__":
    main()
