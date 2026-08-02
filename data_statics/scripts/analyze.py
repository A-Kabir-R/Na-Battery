"""
Master statistical-analysis pipeline for the Standard-cycling sodium-ion dataset.

Walks every .ird file under ../.., streams it into memory, and writes:

  tables/
    inventory.csv          - one row per .ird file (condition, cell, visit, test_type, size, path)
    file_statistics.csv    - one row per .ird file (row count, dt, V/I/T/Ah stats)
    cycle_metrics.csv      - one row per detected cycle (Q_ch, Q_dis, CE, energy, T_max ...)
    condition_summary.csv  - one row per condition (aggregate stats)
    degradation_rates.csv  - one row per (condition, cell) with linear-fit fade slope + EOL cycle

Run with:  python analyze.py            # full run (all files)
           python analyze.py --limit 20 # smoke test on the first 20 files
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]           # .../Standard cycling
OUT  = Path(__file__).resolve().parents[2]           # .../data_statics
TABLES = OUT / "tables"
LOGS   = OUT / "logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parsing ---

FILENAME_RE = re.compile(
    r"^(?P<cell>S\d{4}[A-Z0-9]+)_(?P<test>[A-Z]+)_at(?P<rpt_T>-?\d+)degC_V(?P<visit>\d+)_(?P<testid>\d+)\.ird$"
)


def parse_condition(cond: str) -> Dict[str, Any]:
    """DOD100_1C1C_25degC  ->  {DOD:100, C_ch:1, C_dis:1, T:25}
       DOD100_0.5C0.5C_-10degC -> {DOD:100, C_ch:0.5, C_dis:0.5, T:-10}"""
    m = re.match(
        r"DOD(?P<dod>\d+)_(?P<cch>[\d.]+)C(?P<cdis>[\d.]+)C_(?P<T>-?\d+)degC$",
        cond,
    )
    if not m:
        return {"DOD_pct": np.nan, "C_ch": np.nan, "C_dis": np.nan, "T_degC": np.nan}
    return {
        "DOD_pct": int(m["dod"]),
        "C_ch":    float(m["cch"]),
        "C_dis":   float(m["cdis"]),
        "T_degC":  int(m["T"]),
    }


def find_data_offset(path: Path) -> int:
    """Return number of lines to skip so read_csv lands on the real header row."""
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("###"):
                return i + 1
    return 0


def read_ird(path: Path) -> pd.DataFrame:
    skip = find_data_offset(path)
    df = pd.read_csv(
        path,
        skiprows=skip,
        header=0,
        engine="c",
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    needed = ["Time", "Voltage", "Current", "Ah_counter", "Temperature", "StepID"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {path.name}: have {list(df.columns)}")
    # Time is either a pandas-Timedelta repr ("0 days 00:00:00.000998172") or a number
    t = df["Time"]
    if t.dtype == object:
        df["Time"] = pd.to_timedelta(t, errors="coerce").dt.total_seconds()
    df["Time"] = df["Time"].astype("float64")   # keep 64-bit for accurate dt
    for c in ["Voltage", "Current", "Ah_counter", "Temperature"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["StepID"] = pd.to_numeric(df["StepID"], errors="coerce").fillna(-1).astype("int32")
    return df[needed]


# ---------------------------------------------------------- cycle extraction

def extract_cycles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Segment a CYC file into charge/discharge cycles by rising edges of current sign."""
    t  = df["Time"].to_numpy()
    V  = df["Voltage"].to_numpy()
    I  = df["Current"].to_numpy()
    T  = df["Temperature"].to_numpy()
    Ah = df["Ah_counter"].to_numpy()
    if len(t) < 100:
        return []

    dt = np.diff(t, prepend=t[0])
    # Guard against clock resets / huge idle gaps
    dt = np.clip(dt, 0.0, 300.0)

    # Sign of current with a small dead band
    thr = max(0.05, 0.02 * float(np.nanmax(np.abs(I)) or 0.0))
    sign = np.zeros_like(I, dtype=np.int8)
    sign[I >  thr] =  1
    sign[I < -thr] = -1

    # Cycle = starts on each 0/-1 -> +1 transition
    is_ch = (sign == 1)
    rising = np.where(is_ch[1:] & ~is_ch[:-1])[0] + 1
    if len(rising) < 2:
        return []

    cycles: List[Dict[str, Any]] = []
    for k in range(len(rising) - 1):
        s, e = rising[k], rising[k + 1]
        seg_I  = I[s:e]
        seg_V  = V[s:e]
        seg_T  = T[s:e]
        seg_dt = dt[s:e]
        seg_Ah = Ah[s:e]

        ch = seg_I >  thr
        di = seg_I < -thr
        if not ch.any() or not di.any():
            continue

        Q_ch  = float(np.sum(seg_I[ch] * seg_dt[ch]) / 3600.0)
        Q_dis = float(-np.sum(seg_I[di] * seg_dt[di]) / 3600.0)
        E_ch  = float(np.sum(seg_V[ch] * seg_I[ch] * seg_dt[ch]) / 3600.0)
        E_dis = float(-np.sum(seg_V[di] * seg_I[di] * seg_dt[di]) / 3600.0)

        cycles.append({
            "cycle_in_file":       k + 1,
            "duration_s":          float(t[e - 1] - t[s]),
            "Q_charge_Ah":         Q_ch,
            "Q_discharge_Ah":      Q_dis,
            "coulombic_efficiency": Q_dis / Q_ch if Q_ch > 0 else np.nan,
            "E_charge_Wh":         E_ch,
            "E_discharge_Wh":      E_dis,
            "energy_efficiency":   E_dis / E_ch if E_ch > 0 else np.nan,
            "V_max":               float(np.nanmax(seg_V)),
            "V_min":               float(np.nanmin(seg_V)),
            "T_max":               float(np.nanmax(seg_T)),
            "T_min":               float(np.nanmin(seg_T)),
            "T_mean":              float(np.nanmean(seg_T)),
            "Ah_span":             float(np.nanmax(seg_Ah) - np.nanmin(seg_Ah)),
        })
    return cycles


# --------------------------------------------------------- per-file summary

def file_summary(df: pd.DataFrame) -> Dict[str, Any]:
    t = df["Time"].to_numpy()
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 300)]
    V = df["Voltage"].to_numpy()
    I = df["Current"].to_numpy()
    T = df["Temperature"].to_numpy()
    Ah = df["Ah_counter"].to_numpy()

    ch_mask = I > 0.05
    di_mask = I < -0.05
    dt_full = np.diff(t, prepend=t[0])
    dt_full = np.clip(dt_full, 0.0, 300.0)

    return {
        "rows":            int(len(df)),
        "duration_s":      float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "dt_median_s":     float(np.median(dt)) if dt.size else np.nan,
        "dt_mean_s":       float(np.mean(dt))   if dt.size else np.nan,
        "V_min":  float(np.nanmin(V)), "V_max": float(np.nanmax(V)),
        "V_mean": float(np.nanmean(V)),"V_std": float(np.nanstd(V)),
        "I_min":  float(np.nanmin(I)), "I_max": float(np.nanmax(I)),
        "I_mean": float(np.nanmean(I)),"I_std": float(np.nanstd(I)),
        "T_min":  float(np.nanmin(T)), "T_max": float(np.nanmax(T)),
        "T_mean": float(np.nanmean(T)),"T_std": float(np.nanstd(T)),
        "Ah_min": float(np.nanmin(Ah)),"Ah_max": float(np.nanmax(Ah)),
        "Ah_span":float(np.nanmax(Ah) - np.nanmin(Ah)),
        "n_steps":int(df["StepID"].nunique()),
        "Q_charge_Ah_total":    float(np.sum(I[ch_mask] * dt_full[ch_mask]) / 3600.0),
        "Q_discharge_Ah_total": float(-np.sum(I[di_mask] * dt_full[di_mask]) / 3600.0),
    }


# ------------------------------------------------------------ file walker

def walk_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir() or not cond_dir.name.startswith("DOD"):
            continue
        cond = cond_dir.name
        for sub in cond_dir.iterdir():
            if not sub.is_dir():
                continue
            test_type = "CU" if sub.name.startswith("CU") else ("CYC" if "Cycling" in sub.name else "OTHER")
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() != ".ird":
                    continue
                m = FILENAME_RE.match(f.name)
                if not m:
                    continue
                rows.append({
                    "path":       str(f),
                    "condition":  cond,
                    "test_type":  test_type,
                    "cell":       m["cell"],
                    "visit":      int(m["visit"]),
                    "test_id":    int(m["testid"]),
                    "rpt_T_degC": int(m["rpt_T"]),
                    "size_MB":    round(f.stat().st_size / 1e6, 3),
                })
    inv = pd.DataFrame(rows)
    # attach condition-decoded columns
    dec = inv["condition"].map(parse_condition).apply(pd.Series)
    inv = pd.concat([inv, dec], axis=1)
    return inv


# ------------------------------------------------------- per-file worker

def process_file(row: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(row["path"])
    out = dict(row)
    try:
        df = read_ird(path)
        summ = file_summary(df)
        out.update(summ)
        out["parse_status"] = "ok"
        if row["test_type"] == "CYC":
            out["_cycles"] = extract_cycles(df)
        else:
            out["_cycles"] = []
    except Exception as e:
        out["parse_status"] = f"error: {type(e).__name__}: {e}"
        out["_cycles"] = []
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",   type=int, default=0, help="process only first N files (0 = all)")
    ap.add_argument("--workers", type=int, default=max(2, min(6, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()

    print(f"[analyze] root = {ROOT}")
    print(f"[analyze] out  = {OUT}")
    print(f"[analyze] walking dataset ...", flush=True)
    inv = walk_dataset(ROOT)
    inv.to_csv(TABLES / "inventory.csv", index=False)
    print(f"[analyze] inventory: {len(inv)} files across {inv['condition'].nunique()} conditions")

    if args.limit:
        inv_run = inv.head(args.limit).copy()
        print(f"[analyze] LIMIT applied -> {len(inv_run)} files")
    else:
        inv_run = inv

    tasks = inv_run.to_dict(orient="records")
    n = len(tasks)
    print(f"[analyze] processing {n} files with {args.workers} workers ...", flush=True)

    stats_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(process_file, tasks, chunksize=1):
            done += 1
            cyc = res.pop("_cycles", [])
            stats_rows.append(res)
            for c in cyc:
                c2 = {
                    "condition": res["condition"], "cell": res["cell"],
                    "visit":     res["visit"],     "test_type": res["test_type"],
                    "path":      res["path"],
                    **c,
                }
                cycle_rows.append(c2)
            if done % 20 == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done)
                print(f"[analyze]   {done}/{n}  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(TABLES / "file_statistics.csv", index=False)
    print(f"[analyze] wrote {TABLES/'file_statistics.csv'}  ({len(stats)} rows)")

    cyc_df = pd.DataFrame(cycle_rows)
    cyc_df.to_csv(TABLES / "cycle_metrics.csv", index=False)
    print(f"[analyze] wrote {TABLES/'cycle_metrics.csv'}  ({len(cyc_df)} rows)")

    # ---------------- per-cell global cycle index + degradation summary
    if len(cyc_df):
        cyc_df = cyc_df.sort_values(["condition", "cell", "visit", "cycle_in_file"]).reset_index(drop=True)
        cyc_df["global_cycle"] = cyc_df.groupby(["condition", "cell"]).cumcount() + 1
        cyc_df["cumulative_Ah_throughput"] = cyc_df.groupby(["condition", "cell"])["Q_discharge_Ah"].cumsum()

        # SOH per cell = Q_dis / initial Q_dis (median of first 3 cycles)
        def _soh(g: pd.DataFrame) -> pd.DataFrame:
            head = g.head(3)["Q_discharge_Ah"].median()
            g = g.copy()
            g["Q_initial_Ah"] = head
            g["SOH_pct"] = 100.0 * g["Q_discharge_Ah"] / head if head and head > 0 else np.nan
            return g

        cyc_df = cyc_df.groupby(["condition", "cell"], group_keys=False).apply(_soh)
        cyc_df.to_csv(TABLES / "cycle_metrics.csv", index=False)   # overwrite w/ enriched

        # Per-(condition, cell) degradation: linear slope of SOH vs cycle + EOL cycle (< 80%)
        deg_rows = []
        for (cond, cell), g in cyc_df.groupby(["condition", "cell"]):
            g = g.dropna(subset=["SOH_pct"])
            if len(g) < 5:
                continue
            x = g["global_cycle"].to_numpy(dtype=float)
            y = g["SOH_pct"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            below = g[g["SOH_pct"] < 80]
            eol_cycle = int(below["global_cycle"].iloc[0]) if len(below) else np.nan
            deg_rows.append({
                "condition": cond, "cell": cell,
                "n_cycles":  int(len(g)),
                "Q_initial_Ah": float(g["Q_initial_Ah"].iloc[0]),
                "Q_final_Ah":   float(g["Q_discharge_Ah"].iloc[-1]),
                "SOH_final_pct": float(g["SOH_pct"].iloc[-1]),
                "fade_slope_pct_per_cycle": float(slope),
                "fade_intercept_pct":       float(intercept),
                "eol80_cycle":              eol_cycle,
                "cumulative_Ah_final":      float(g["cumulative_Ah_throughput"].iloc[-1]),
                "CE_median":                float(g["coulombic_efficiency"].median()),
                "EE_median":                float(g["energy_efficiency"].median()),
            })
        deg = pd.DataFrame(deg_rows)
        for c in ["condition"]:
            deg = deg.merge(inv[[c] + ["DOD_pct", "C_ch", "C_dis", "T_degC"]].drop_duplicates(),
                            on=c, how="left")
        deg.to_csv(TABLES / "degradation_rates.csv", index=False)
        print(f"[analyze] wrote {TABLES/'degradation_rates.csv'}  ({len(deg)} rows)")

        # Condition-level summary (mean across cells)
        cond_sum = (deg.groupby("condition")
                       .agg(cells=("cell", "nunique"),
                            n_cycles_mean=("n_cycles", "mean"),
                            Q_initial_Ah=("Q_initial_Ah", "mean"),
                            SOH_final_pct=("SOH_final_pct", "mean"),
                            fade_slope=("fade_slope_pct_per_cycle", "mean"),
                            eol80_cycle=("eol80_cycle", "mean"),
                            CE_median=("CE_median", "mean"),
                            EE_median=("EE_median", "mean"))
                       .reset_index())
        cond_sum = cond_sum.merge(inv[["condition", "DOD_pct", "C_ch", "C_dis", "T_degC"]].drop_duplicates(),
                                   on="condition", how="left")
        cond_sum.to_csv(TABLES / "condition_summary.csv", index=False)
        print(f"[analyze] wrote {TABLES/'condition_summary.csv'}  ({len(cond_sum)} rows)")

    # ---------------- parse error log
    errs = stats[stats["parse_status"] != "ok"]
    if len(errs):
        errs[["path", "parse_status"]].to_csv(LOGS / "parse_errors.csv", index=False)
        print(f"[analyze] {len(errs)} parse errors -> {LOGS/'parse_errors.csv'}")

    print(f"[analyze] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
