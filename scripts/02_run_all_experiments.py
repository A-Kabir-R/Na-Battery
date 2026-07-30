"""Run every (preprocessing × target × model) combination on GroupKFold splits.

Writes long-format results to the configured results directory
(``artifacts/results/[experiment.results_subdir/]raw_results.csv``).
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.io.loaders import load_config
from src.models.registry import enabled_classifiers, enabled_regressors
from src.pipeline.run_experiment import (CLASSIFICATION_TARGETS,
                                          REGRESSION_TARGETS, run_one)
from src.splits.group_kfold import (
    build_locked_split_manifest,
    validate_locked_split_manifest,
)
from src.utils.progress import tqdm_joblib

#: Fallback only. The active list comes from
#: ``experiment.preprocessing_levels`` in config.yaml, which is ["unified"].
#: P1/P2/P3 remain loadable for traceability but are not part of the final
#: experiment.
DEFAULT_PIPELINES = ["unified"]
REG_TARGETS = list(REGRESSION_TARGETS)
CLF_TARGETS = list(CLASSIFICATION_TARGETS)


def _load_pipeline(name: str, features_dir: Path) -> pd.DataFrame:
    p = features_dir / f"{name}.parquet"
    return pd.read_parquet(p)


def _results_dir(cfg: dict) -> Path:
    base = Path(cfg["paths"]["artifacts"]) / "results"
    subdir = (cfg.get("experiment") or {}).get("results_subdir") or ""
    subdir = str(subdir).strip()
    return base / subdir if subdir else base


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the planned (preprocessing, target, model) combinations without fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    results_dir = _results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["split"]["n_splits"]
    rs = cfg["split"]["random_state"]
    workers = int(cfg["experiment"]["parallel_workers"])
    model_n_jobs = (cfg.get("experiment") or {}).get("model_n_jobs")
    if model_n_jobs is not None:
        model_n_jobs = int(model_n_jobs)

    # Active preprocessing levels come from config; the final experiment uses
    # only "unified" so every retained model reads one identical feature table.
    pipelines = [
        str(level) for level in
        ((cfg.get("experiment") or {}).get("preprocessing_levels")
         or DEFAULT_PIPELINES)
    ]
    evaluate_holdout = bool(
        (cfg.get("experiment") or {}).get("evaluate_holdout", False)
    )

    reg_models = enabled_regressors(cfg, model_n_jobs=model_n_jobs)
    clf_models = enabled_classifiers(cfg, model_n_jobs=model_n_jobs) if CLF_TARGETS else {}

    reg_names = list(reg_models.keys())
    clf_names = list(clf_models.keys())
    expected_reg_combos = len(pipelines) * len(REG_TARGETS) * len(reg_names)
    expected_clf_combos = len(pipelines) * len(CLF_TARGETS) * len(clf_names)

    print(f"[run] enabled regressors: {', '.join(reg_names)}", flush=True)
    if clf_names:
        print(f"[run] enabled classifiers: {', '.join(clf_names)}", flush=True)
    print(f"[run] preprocessing pipelines: {', '.join(pipelines)}", flush=True)
    print(f"[run] regression targets: {len(REG_TARGETS)}", flush=True)
    print(f"[run] expected experiment combinations: {expected_reg_combos + expected_clf_combos}",
          flush=True)
    print(f"[run] results directory: {results_dir}", flush=True)
    print(f"[run] locked-holdout evaluation: "
          f"{'ENABLED' if evaluate_holdout else 'disabled (development only)'}",
          flush=True)
    print(f"[run] outer workers: {workers}  |  model n_jobs: {model_n_jobs}", flush=True)

    if args.dry_run:
        print("[run] --dry-run: listing planned combinations only", flush=True)
        for pipe, target, name in product(pipelines, REG_TARGETS, reg_names):
            print(f"[run]   {pipe:>3} | {target:>25} | {name}", flush=True)
        for pipe, target, name in product(pipelines, CLF_TARGETS, clf_names):
            print(f"[run]   {pipe:>3} | {target:>25} | {name}", flush=True)
        return

    reference = _load_pipeline(pipelines[0], features_dir)
    primary_target = REG_TARGETS[0]
    reference = reference.dropna(subset=[primary_target]).reset_index(drop=True)
    split_dir = Path(cfg["paths"]["artifacts"]) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split_manifest.parquet"
    if split_path.exists():
        split_manifest = pd.read_parquet(split_path)
        validate_locked_split_manifest(
            reference, split_manifest, n_splits=n_splits,
            holdout_fraction=float(cfg["split"].get("holdout_fraction", 0.25)),
            random_state=rs,
        )
        if set(reference["cell"].astype(str).unique()) != set(split_manifest["cell"].astype(str)):
            raise RuntimeError(
                "locked split cells differ from the current primary-target cohort; "
                "archive the old artifacts and remove splits/split_manifest.parquet to regenerate"
            )
    else:
        split_manifest = build_locked_split_manifest(
            reference,
            holdout_fraction=float(cfg["split"].get("holdout_fraction", 0.25)),
            n_splits=n_splits,
            stratify_col=str(cfg["split"].get("stratify_by", "condition")),
            random_state=rs,
            holdout_candidates=int(cfg["split"].get("holdout_candidates", 1024)),
        )
        temporary = split_path.with_suffix(".parquet.tmp")
        split_manifest.to_parquet(temporary, index=False)
        temporary.replace(split_path)
    split_manifest.to_csv(split_dir / "split_manifest.csv", index=False)
    print(
        "[run] locked split: "
        f"development={int((split_manifest['outer_role'] == 'development').sum())} cells, "
        f"holdout={int((split_manifest['outer_role'] == 'holdout').sum())} cells, "
        f"cv_folds={n_splits}"
    )

    all_rows = []
    total_t = time.time()

    outer = tqdm(pipelines, desc="[run] pipelines", unit="pipe")
    for pipe in outer:
        outer.set_postfix_str(pipe)
        try:
            df = _load_pipeline(pipe, features_dir)
        except FileNotFoundError:
            tqdm.write(f"[run] {pipe}.parquet missing — skipping")
            continue
        tqdm.write(f"[run] {pipe}: {df.shape}")

        reg_jobs = list(product(REG_TARGETS, reg_models.items()))
        clf_jobs = list(product(CLF_TARGETS, clf_models.items()))

        def _job(pipe, target, name, model, kind):
            t0 = time.time()
            rows = run_one(df, pipe, target, name, model, kind=kind,
                            n_splits=n_splits, random_state=rs,
                            split_manifest=split_manifest,
                            evaluate_holdout=evaluate_holdout)
            tqdm.write(f"[run] {pipe:>3} | {target:>15} | {name:<20} | "
                       f"{len(rows):3d} rows | {time.time()-t0:5.1f}s")
            return rows

        if reg_jobs:
            with tqdm_joblib(tqdm(total=len(reg_jobs),
                                  desc=f"[run] {pipe} regressors", unit="fit")):
                res = Parallel(n_jobs=workers, backend="loky")(
                    delayed(_job)(pipe, t, n, m, "regression")
                    for t, (n, m) in reg_jobs
                )
            for r in res:
                all_rows.extend(r)

        if clf_jobs:
            with tqdm_joblib(tqdm(total=len(clf_jobs),
                                  desc=f"[run] {pipe} classifiers", unit="fit")):
                res = Parallel(n_jobs=workers, backend="loky")(
                    delayed(_job)(pipe, t, n, m, "classification")
                    for t, (n, m) in clf_jobs
                )
            for r in res:
                all_rows.extend(r)
    outer.close()

    raw = pd.DataFrame(all_rows)
    out = results_dir / "raw_results.csv"
    temporary = out.with_suffix(".csv.tmp")
    raw.to_csv(temporary, index=False)
    temporary.replace(out)
    print(f"[run] wrote {out} ({len(raw)} rows) in {time.time()-total_t:.1f}s")

    if not raw.empty:
        present = set(raw["model"].astype(str).unique())
        allowed = set(reg_names) | set(clf_names)
        unexpected = sorted(present - allowed)
        if unexpected:
            raise SystemExit(
                f"raw_results.csv contains models outside the allowlist: {unexpected}"
            )

    failed = raw[raw["status"].ne("completed")] if not raw.empty else raw
    if raw.empty or not failed.empty:
        if not failed.empty:
            print("[run] failed experiment partitions:")
            print(failed[[
                "preprocessing", "target", "model", "fold", "error_message",
            ]].drop_duplicates().to_string(index=False))
        raise SystemExit("experiment stage incomplete; refusing aggregation and publication plots")


if __name__ == "__main__":
    main()
