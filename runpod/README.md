# RunPod deployment scripts

End-to-end flow: **push code -> set up pod (deps + raw dataset) -> build canonical tables -> run the ML pipeline -> upload results to R2 -> self-terminate -> pull results locally.**

## Files

| Script | Where you run it | What it does |
| --- | --- | --- |
| `r2.env.example` | — | Template. Copy to `r2.env` and fill in R2 + RunPod + SSH creds and the pod layout (`POD_ROOT`, `POD_CODE_DIR`, `POD_DATA_STATICS_DIR`, `POD_DATASET_DIR`). |
| `push.sh` | **Laptop** | Rsyncs the complete source tree to `$POD_CODE_DIR`, validates required canonical-pipeline files, and optionally ships legacy data-statistics tables. `--with-env` also pushes `r2.env`. |
| `setup.sh` | **Pod** | apt packages, uv + venv, `pip install -r requirements.txt`, downloads + extracts the RWTH dataset zip into `$POD_DATASET_DIR`, auto-detects the `DODxxx_*` subtree and writes `runpod/pod.env` with the resolved `SIB_DATASET_ROOT`. Idempotent. |
| `run.sh` | **Pod** | Sources `r2.env` + `pod.env`, builds missing canonical prerequisites, runs the requested stages, bundles the configured artifact and log directories, uploads to R2, and honours `STOP_MODE`. |
| `pull.sh` | **Laptop** | Downloads bundle(s) from R2 into `runpod/downloads/`. |

## Data flow

The model pipeline no longer depends on laptop-generated cycle CSVs. The
`canonical` stage parses the raw Standard-cycling IRD files and writes the file
manifest, steps, cycles, RPT measurements, metadata, and QC tables under
`$SIB_CANONICAL`.

## First-time setup

```bash
cp runpod/r2.env.example runpod/r2.env
# edit runpod/r2.env — fill R2_*, RUNPOD_API_KEY, RUNPOD_SSH_* fields.
chmod +x runpod/*.sh
```

## Typical run

```bash
# --- on laptop ---
./runpod/push.sh --with-env               # ships code + data statics/tables + r2.env

# --- on pod (SSH in, use tmux so the run survives disconnects) ---
tmux new -s sib
cd /workspace/sib/code
bash runpod/setup.sh                      # ~5-15 min: deps + dataset download + probe
bash runpod/run.sh 2>&1 | tee run.log
# detach with Ctrl-b d ; run.sh self-terminates the pod on success

# --- on laptop, once the pod has terminated ---
./runpod/pull.sh                          # newest bundle -> runpod/downloads/
```

## Pipeline stages

`run.sh` honours the `RUN_STAGES` env var (comma-separated). Valid stages:

| stage | script | approx runtime |
| --- | --- | --- |
| `smoke` | `scripts/00_smoke_test.py` | <2 min |
| `canonical` | `scripts/00_build_canonical.py` | depends on raw storage and checksum speed |
| `build` | `scripts/01_build_features.py` | ~30–60 min (P3 raw-`.ird` re-parse dominates) |
| `degradation` | `scripts/05_analyze_degradation.py` | minutes |
| `experiments` | `scripts/02_run_all_experiments.py` | ~15–45 min depending on `experiment.parallel_workers` |
| `aggregate` | `scripts/03_aggregate_results.py` | minutes (cell bootstrap dominates) |
| `plot` | `scripts/04_plot_results.py` | minutes (PNG and PDF publication suite) |

Default: `canonical,build,degradation,experiments,aggregate,plot`. If an older `r2.env`
still starts with `build`, `run.sh` automatically runs `canonical` first when
its required tables are missing.
Quick smoke run on the pod: `RUN_STAGES=smoke bash runpod/run.sh`.

## Path resolution

`config.yaml` uses `${env:VAR:-default}`. On the pod, `run.sh` exports:

```
SIB_ROOT          -> $POD_ROOT             (e.g. /workspace/sib)
SIB_DATA_STATICS  -> $POD_DATA_STATICS_DIR (e.g. /workspace/sib/data statics)
SIB_DATASET_ROOT  -> auto-detected DOD subtree from the extracted zip
SIB_ARTIFACTS     -> $POD_RESULTS_DIR      (e.g. /workspace/sib/code/artifacts)
SIB_CANONICAL     -> $POD_RESULTS_DIR/canonical
SIB_LOGS          -> $POD_CODE_DIR/logs
```

On the laptop none of these vars are set, so the `default` half of each `${env:...:-default}` kicks in and the existing local paths are used. No code changes needed to switch between the two.

## Safety behaviour of `run.sh`

- If any stage exits non-zero → later stages are skipped, pod is **not** shut down (so you can inspect), but we still try to upload whatever's on disk.
- If R2 upload fails → **no shutdown**, so the bundle isn't lost.
- Otherwise honours `STOP_MODE` (`terminate` / `stop` / `none`).

## What lands in R2

- `s3://$R2_BUCKET/$R2_PREFIX/results_<UTC>.tar.zst` — complete auditable bundle of canonical tables, feature matrices, locked splits, models, predictions, metrics, publication plots, and logs.
- `s3://$R2_BUCKET/$R2_PREFIX/results_<UTC>.meta.json` — sidecar with pipeline rc, pod id, bundle size, stages run.

## Troubleshooting

- **`push.sh` prompts for a password** — set `RUNPOD_SSH_KEY` to the private key that matches the pubkey you added in RunPod → Settings → SSH Public Keys.
- **`run.sh` says pod.env missing** — run `setup.sh` first (it writes `runpod/pod.env`).
- **`01_build_features.py` says canonical tables are missing** — deploy the latest `run.sh`; it automatically runs `scripts/00_build_canonical.py` before `build`.
- **Import fails for `protocol_segmentation` or `canonical`** — the pod has a partial source deployment. Re-run `./runpod/push.sh --code-only`; the push now validates these files remotely.
- **P3 stage says "no CYC files found"** — `SIB_DATASET_ROOT` didn't resolve to the folder that contains `DOD*_*C*C_*degC/`. Check `runpod/pod.env` and the `ls` output at the end of `setup.sh`.
- **`aws: command not found`** on the pod — re-run `setup.sh` (it installs `awscli` into the venv).
