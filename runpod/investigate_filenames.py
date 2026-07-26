#!/usr/bin/env python3
"""Diagnose why only cell V00 gets visit=0 records.

Runs against preprocessing/manifests/file_index.parquet on the pod:

    cd /workspace/sodium_ion_battery_project
    source .venv/bin/activate
    python runpod/investigate_filenames.py

Prints:
  1. file_index inventory (rows, columns, null counts)
  2. visit_from_filename distribution overall + per cell
  3. For each cell: unique visit numbers observed in its filenames
  4. Files with NaN visit_from_filename — sampled filenames per cell
  5. Files where visit==0 (the ones that DO match) — sampled filenames per cell
  6. Cross-check: cell_id_from_filename vs cell_id_from_header agreement
  7. header_start_time coverage (potential alternative for ordering)
  8. Concrete recommendation on how to derive `visit` correctly
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120)


def _hdr(t: str) -> None:
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def main() -> int:
    root = Path(os.environ.get("SIB_PROJECT_ROOT", "/workspace/results"))
    fi_path = root / "preprocessing" / "manifests" / "file_index.parquet"
    print(f"[investigate] file_index: {fi_path}")

    if not fi_path.exists():
        print(f"MISSING: {fi_path}")
        return 1

    fi = pd.read_parquet(fi_path)
    print(f"loaded {len(fi)} file rows")

    # ---------------------------------------------------------------- 1
    _hdr("1. file_index — inventory")
    key_cols = ["filename", "cell_id_from_filename", "cell_id_from_header",
                "visit_from_filename", "test_id", "filename_file_type",
                "header_test_name", "header_start_time",
                "parent_condition", "modeling_scope"]
    key_cols = [c for c in key_cols if c in fi.columns]
    print(f"columns ({len(fi.columns)}): {list(fi.columns)}")
    print("\nnull counts (key columns):")
    print(fi[key_cols].isna().sum().to_string())

    # canonical cell id: prefer header, fall back to filename
    cid = fi["cell_id_from_header"].where(
        fi["cell_id_from_header"].notna() & (fi["cell_id_from_header"].astype(str) != ""),
        fi["cell_id_from_filename"])
    fi = fi.assign(_cell_id=cid.fillna("").astype(str))

    # ---------------------------------------------------------------- 2
    _hdr("2. visit_from_filename distribution overall")
    vfilm = fi["visit_from_filename"]
    print(f"total files: {len(fi)}")
    print(f"NaN visit_from_filename: {int(vfilm.isna().sum())}")
    print("value counts (non-NaN):")
    print(vfilm.dropna().astype(int).value_counts().sort_index().to_string())

    # ---------------------------------------------------------------- 3
    _hdr("3. per-cell: unique visits observed in filenames")
    rows = []
    for cid_val, g in fi.groupby("_cell_id"):
        v = g["visit_from_filename"].dropna().astype(int)
        rows.append({
            "cell_id": cid_val or "<blank>",
            "n_files": len(g),
            "n_visits_named": len(v),
            "n_no_visit": int(g["visit_from_filename"].isna().sum()),
            "unique_visits": sorted(v.unique().tolist()),
        })
    per_cell = pd.DataFrame(rows).sort_values("cell_id")
    # Truncate unique_visits printout
    per_cell["unique_visits"] = per_cell["unique_visits"].apply(
        lambda lst: str(lst) if len(lst) <= 12 else f"{lst[:6]}...{lst[-3:]} (n={len(lst)})")
    print(per_cell.to_string(index=False))

    # ---------------------------------------------------------------- 4
    _hdr("4. files with NaN visit_from_filename — sample filenames per cell")
    nan_v = fi[fi["visit_from_filename"].isna()]
    print(f"total: {len(nan_v)} files")
    if len(nan_v):
        for cid_val, g in nan_v.groupby("_cell_id"):
            print(f"\ncell={cid_val or '<blank>'}  ({len(g)} files)")
            sample = g["filename"].head(6).tolist()
            for fn in sample:
                print(f"  {fn}")
            if len(g) > 6:
                print(f"  ... ({len(g) - 6} more)")

    # ---------------------------------------------------------------- 5
    _hdr("5. files where visit_from_filename == 0 — the ONLY ones giving V00 anchors")
    v0 = fi[fi["visit_from_filename"] == 0]
    print(f"total: {len(v0)} files across cells: {sorted(v0['_cell_id'].unique().tolist())}")
    for cid_val, g in v0.groupby("_cell_id"):
        print(f"\ncell={cid_val}  ({len(g)} files)")
        for fn in g["filename"].head(8):
            print(f"  {fn}")
        if len(g) > 8:
            print(f"  ... ({len(g) - 8} more)")

    # ---------------------------------------------------------------- 6
    _hdr("6. cell_id agreement: filename vs header")
    a = fi[["cell_id_from_filename", "cell_id_from_header"]].copy()
    a = a.fillna("").astype(str)
    a["match"] = a["cell_id_from_filename"] == a["cell_id_from_header"]
    print(f"agree     : {int(a['match'].sum())}")
    print(f"disagree  : {int((~a['match']).sum())}")
    dis = a[~a["match"]].groupby(["cell_id_from_filename", "cell_id_from_header"]).size()
    dis = dis.reset_index(name="n").sort_values("n", ascending=False).head(20)
    print("\ntop mismatches:")
    print(dis.to_string(index=False))

    # ---------------------------------------------------------------- 7
    _hdr("7. header_start_time coverage (potential visit-ordering signal)")
    if "header_start_time" in fi.columns:
        hst = fi["header_start_time"]
        print(f"non-null: {int(hst.notna().sum())}/{len(fi)}")
        print("\nsample values (first 10 unique):")
        for v in hst.dropna().astype(str).unique()[:10]:
            print(f"  {v!r}")
    else:
        print("no header_start_time column")

    # ---------------------------------------------------------------- 8
    _hdr("8. regex live-check on a sample of filenames")
    _V = re.compile(r"[_-]V(\d{2,3})", re.IGNORECASE)
    # Alternative regexes worth considering
    alts = {
        "current  [_-]V(\\d{2,3})": re.compile(r"[_-]V(\d{2,3})", re.IGNORECASE),
        "start   ^V(\\d{2,3})[_-]": re.compile(r"^V(\d{2,3})[_-]", re.IGNORECASE),
        "loose  (?<![A-Za-z])V(\\d{2,3})(?![A-Za-z0-9])": re.compile(
            r"(?<![A-Za-z])V(\d{2,3})(?![A-Za-z0-9])", re.IGNORECASE),
    }
    sample = fi["filename"].dropna().unique().tolist()
    print(f"unique filenames in index: {len(sample)}")
    for name, pat in alts.items():
        hits = sum(1 for fn in sample if pat.search(fn))
        print(f"  {name!s:<55} matches {hits}/{len(sample)}")

    print("\nExample filenames from cells with no visit=0 anchor (V01, V02, V04):")
    for cid_target in ["V01", "V02", "V04"]:
        g = fi[fi["_cell_id"] == cid_target].sort_values("filename")
        if not len(g):
            continue
        print(f"\n  cell={cid_target}:")
        for fn in g["filename"].head(8):
            m_cur = _V.search(fn)
            print(f"    {fn}   -> current-regex match: {m_cur.group(0) if m_cur else None}")

    _hdr("done — how to derive `visit` correctly")
    print(dedent := """\
Based on the output above, choose the fix:

 (a) If sections 4+5 show most files have NO '_V##' at all, and cell V00's
     files just happen to contain '_V00' (e.g. a subtest labeled V00 that
     other cells don't have), then the regex is the wrong tool. Fix:
     assign `visit` by ordering per-cell characterization files by
     `header_start_time` (or file mtime), then rank 0..N.

 (b) If sections 4+5 show other cells DO have '_V##' markers but the regex
     misses them (loose-regex hit count in section 8 is much higher than
     current-regex hit count), tighten/relax _VISIT_RE accordingly.

 (c) If both patterns coexist, use a two-tier strategy: prefer the regex
     match when present, fall back to per-cell time-order rank.
""")
    print(dedent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
