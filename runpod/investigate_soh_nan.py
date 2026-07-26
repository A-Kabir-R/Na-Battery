#!/usr/bin/env python3
"""On-pod diagnostic for the T6 knee ValueError and upstream SOH_pct coverage.

Run on the pod (results already on disk, no R2 round-trip needed):

    cd /workspace/sodium_ion_battery_project
    source .venv/bin/activate
    python runpod/investigate_soh_nan.py

Or with a custom results root:

    SIB_PROJECT_ROOT=/workspace/results python runpod/investigate_soh_nan.py

Prints (in order):
  1. Capacity file inventory + null counts
  2. Per-cell V00 anchor status (the reason SOH_pct is NaN for most cells)
  3. T6 input set: what build_t6_knee actually sees per cell
  4. Whether the fix will yield a non-empty T6_knee.parquet
  5. Cyc_blocks coverage cross-check (explains the T5 warnings too)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)


def _hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _load(p: Path) -> pd.DataFrame:
    if not p.exists():
        print(f"  MISSING: {p}")
        return pd.DataFrame()
    return pd.read_parquet(p)


def main() -> int:
    root = Path(os.environ.get("SIB_PROJECT_ROOT", "/workspace/results"))
    pre = root / "preprocessing"
    print(f"[investigate] SIB_PROJECT_ROOT = {root}")
    print(f"[investigate] preprocessing dir = {pre}")

    cap = _load(pre / "canonical" / "capacity_soh.parquet")
    blocks = _load(pre / "canonical" / "cyc_blocks.parquet")
    t6_out = _load(pre / "canonical" / "targets" / "T6_knee.parquet")  # may not exist yet

    if cap.empty:
        print("\ncapacity_soh.parquet is missing/empty — cannot diagnose SOH.")
        print(f"expected at: {pre / 'canonical' / 'capacity_soh.parquet'}")
        print("contents of preprocessing/canonical/:")
        canonical = pre / "canonical"
        if canonical.exists():
            for p in sorted(canonical.rglob("*")):
                print(f"  {p.relative_to(pre)}")
        else:
            print(f"  (directory does not exist)")
        return 1

    # -----------------------------------------------------------------------
    _hdr("1. capacity.parquet — inventory")
    print(f"total rows: {len(cap)}")
    print(f"columns   : {list(cap.columns)}")
    print("\nrole × signal_validated:")
    print(pd.crosstab(cap["role"], cap["signal_validated"], margins=True))
    print("\nnull counts (key columns):")
    keys = [c for c in ["SOH_pct", "Q_delta_Ah", "reference_Q_Ah",
                        "signal_validated", "visit", "cell_id", "role"]
            if c in cap.columns]
    print(cap[keys].isna().sum().to_string())

    # -----------------------------------------------------------------------
    _hdr("2. per-cell V00 reference_discharge status (root cause of NaN SOH)")
    ref = cap[cap["role"] == "reference_discharge"].copy()
    if ref.empty:
        print("no reference_discharge rows at all — capacity extraction is broken upstream")
    else:
        # Per cell: does a validated V00 exist?
        v00 = (ref[(ref["visit"] == 0) & ref["signal_validated"]]
               .groupby("cell_id")["Q_delta_Ah"].first())
        cells = sorted(ref["cell_id"].dropna().unique().tolist())
        rows = []
        for cid in cells:
            g = ref[ref["cell_id"] == cid]
            q_ref = v00.get(cid, np.nan)
            rows.append({
                "cell_id": cid,
                "n_ref_rows": int(len(g)),
                "n_validated": int(g["signal_validated"].sum()),
                "has_v00_validated": bool(cid in v00.index),
                "v00_Q_Ah": None if pd.isna(q_ref) else round(float(q_ref), 4),
                "v00_usable": bool(np.isfinite(q_ref) and q_ref > 0),
                "n_soh_nonNaN": int(g["SOH_pct"].notna().sum())
                                 if "SOH_pct" in g.columns else 0,
            })
        per_cell = pd.DataFrame(rows)
        print(per_cell.to_string(index=False))
        print(f"\ncells with usable V00 anchor: {int(per_cell['v00_usable'].sum())} / {len(per_cell)}")
        print(f"cells with any non-NaN SOH_pct: {int((per_cell['n_soh_nonNaN']>0).sum())}")

    # -----------------------------------------------------------------------
    _hdr("3. T6 input set — exactly what build_t6_knee sees")
    t6_in = cap[(cap["role"] == "reference_discharge") & cap["signal_validated"]].copy()
    print(f"T6 input rows (role==reference_discharge & signal_validated): {len(t6_in)}")
    if not t6_in.empty and "SOH_pct" in t6_in.columns:
        rows = []
        for cid, g in t6_in.sort_values(["cell_id", "visit"]).groupby("cell_id"):
            soh = g["SOH_pct"].to_numpy(dtype=float)
            n_total = int(len(soh))
            n_valid = int(np.isfinite(soh).sum())
            rows.append({
                "cell_id": cid,
                "n_visits_in_T6": n_total,
                "n_soh_valid": n_valid,
                "will_produce_knee": bool(n_valid >= 5),
                "soh_min": None if n_valid == 0 else round(float(np.nanmin(soh)), 2),
                "soh_max": None if n_valid == 0 else round(float(np.nanmax(soh)), 2),
                "old_would_raise": bool(n_total >= 5 and n_valid == 0),
            })
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        n_bad_before = int(summary["old_would_raise"].sum())
        n_good_after = int(summary["will_produce_knee"].sum())
        print(f"\ncells that would trigger the ValueError under the OLD code: {n_bad_before}")
        print(f"cells that will produce a knee with the NEW code           : {n_good_after}")

    # -----------------------------------------------------------------------
    _hdr("4. existing T6_knee.parquet (if pipeline reran after fix)")
    if t6_out.empty:
        print("T6_knee.parquet not present — pipeline hasn't completed T6 with the fix yet")
    else:
        print(t6_out.to_string(index=False))

    # -----------------------------------------------------------------------
    _hdr("5. cyc_blocks EFC coverage (context for T5 fallback warnings)")
    if blocks.empty:
        print("cyc_blocks.parquet missing")
    else:
        if "cell_id" in blocks.columns:
            per_cell_blocks = blocks.groupby("cell_id").size().rename("n_blocks")
            all_cells = sorted(set(cap["cell_id"].dropna().unique().tolist())
                                | set(blocks["cell_id"].dropna().unique().tolist()))
            cov = pd.DataFrame(index=all_cells)
            cov["in_capacity"] = cov.index.isin(cap["cell_id"].dropna().unique())
            cov["cyc_blocks_rows"] = per_cell_blocks.reindex(cov.index).fillna(0).astype(int)
            print(cov.to_string())
            n_missing = int((cov["cyc_blocks_rows"] == 0).sum())
            print(f"\ncells with zero cyc_blocks rows: {n_missing} / {len(cov)}"
                  f"  (these fall back to visit_index * 25 EFC in T5)")
        else:
            print(f"cyc_blocks has {len(blocks)} rows, columns: {list(blocks.columns)}")

    _hdr("done")
    print("Interpretation guide:")
    print(" - If section 2 shows only 1 cell with a usable V00 anchor, that's the")
    print("   root cause of most cells having all-NaN SOH_pct. The T6 fix drops")
    print("   those cells silently — T6 output will contain only cells with V00.")
    print(" - If section 3 shows 'old_would_raise' > 0, the fix was necessary.")
    print(" - If section 3 shows 'will_produce_knee' == 0, T6 output will be empty")
    print("   — the pipeline will complete but T6 provides no signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
