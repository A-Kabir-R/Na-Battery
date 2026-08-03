# Sodium-Ion Battery Standard-Cycling Pipeline

This repository builds validated canonical tables directly from the RWTH
Standard-cycling `.ird` files, then constructs leakage-safe next-RPT anchors and
cell-grouped model experiments.

The legacy precomputed `data statics/tables/cycle_metrics.csv` is no longer an
accepted model input. Existing model results must not be interpreted after the
cycle and target defects documented in `docs/AUDIT_AND_MIGRATION.md`.

## Current Safety Gates

- The dataset root must directly contain the Standard-cycling `DOD*` condition folders.
- File boundaries and unmatched phases remain explicit incomplete-cycle records.
- CU capacity is one validated full-range 0.5C reference discharge, not total CU throughput.
- One next-RPT target is created per cycling-file anchor.
- A cell-level 25% benchmark holdout is locked before model fitting.
- Five-fold grouped CV, feature pruning, imputation, and scaling use only the 75% development cells.
- Holdout data is never passed as a fitting or early-stopping set; because all
  prespecified models are benchmarked on it, holdout comparisons are explicitly
  labeled exploratory and model ranking uses development OOF CV only.
- RPT capacity, RPT-SOH, and next-minus-previous RPT targets are reported separately.
- Row-pooled, cell-macro, condition-macro, horizon, bootstrap-CI, and plausibility results are persisted.
- Proxy CE/SOH/temperature gassing labels are disabled.
- Legacy online-SOH and uncensored-RUL benchmarks are disabled by default.
- Every fold atomically saves metrics, predictions, model, preprocessor, features, and status.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Override platform paths with `SIB_DATASET_ROOT`, `SIB_ARTIFACTS`, and
`SIB_CANONICAL`. `SIB_DATASET_ROOT` must resolve to the directory directly
containing the Standard-cycling `DOD*` condition folders.

## Run

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/00_smoke_test.py
.venv/bin/python scripts/00_build_canonical.py
.venv/bin/python scripts/01_build_features.py
.venv/bin/python scripts/05_analyze_degradation.py
.venv/bin/python scripts/02_run_all_experiments.py
.venv/bin/python scripts/03_aggregate_results.py
.venv/bin/python scripts/04_plot_results.py
```

Use a separate output for a limited canonical smoke build:

```bash
.venv/bin/python scripts/00_build_canonical.py \
  --limit-files 12 \
  --skip-checksums \
  --output-dir /tmp/na-battery-canonical-smoke
```

Never point a limited run at the canonical production output directory.

## Canonical Outputs

`artifacts/canonical/` contains:

- `file_manifest.parquet`
- `steps.parquet`
- `cycles.parquet`
- `rpt_measurements.parquet`
- `cell_metadata.parquet`
- `events.parquet`
- `qc_flags.parquet`
- `build_manifest.json`
- Optional partitioned `samples/` when `--write-samples` is used

Feature outputs are written to `artifacts/features/`. The locked assignment is
written to `artifacts/splits/split_manifest.parquet`. Development-CV outputs are
under `artifacts/folds/<pipeline>/<target>/<model>/fold_<n>/`; each prespecified
model's development fit and exploratory holdout prediction are under the adjacent
`holdout/` directory. Results
are resumed only when every required artifact exists and its fingerprint matches.

`artifacts/results/` includes prediction-level pooled and macro metrics, grouped
bootstrap intervals, per-cell/per-condition tables, horizon results, prediction
plausibility, feature stability/importance, RPT degradation summaries, and both
vector PDF plus 300-dpi PNG figures (config: `plotting.formats`).

## Scientific Blockers

Before enabling survival or CID early-warning experiments:

1. Confirm the inferred full-range 0.5C RPT step against authoritative protocol metadata.
2. Provide observed CID/gassing outcomes, event times, censoring times, and evidence sources.
3. Approve the sustained capacity-EOL and prediction-horizon definitions.
4. Review all error-severity rows in `qc_flags.parquet`.

No model accuracy should be reported until these decisions and the full-dataset
canonical QC review are complete.
