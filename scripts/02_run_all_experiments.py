"""Run every (preprocessing × target × model) combination on GroupKFold splits.

Writes long-format results to artifacts/results/raw_results.csv.
"""
from __future__ import annotations

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
from src.models.registry import classifiers, regressors
from src.pipeline.run_experiment import (CLASSIFICATION_TARGETS,
                                          REGRESSION_TARGETS, run_one)
from src.splits.group_kfold import (
    build_locked_split_manifest,
    validate_locked_split_manifest,
)
from src.utils.progress import tqdm_joblib

PIPELINES = ["p1", "p2", "p3"]
REG_TARGETS = list(REGRESSION_TARGETS)
CLF_TARGETS = list(CLASSIFICATION_TARGETS)


def _load_pipeline(name: str, features_dir: Path) -> pd.DataFrame:
    p = features_dir / f"{name}.parquet"
    return pd.read_parquet(p)


def main() -> None:
    cfg = load_config()
    features_dir = Path(cfg["paths"]["artifacts"]) / "features"
    results_dir = Path(cfg["paths"]["artifacts"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["split"]["n_splits"]
    rs = cfg["split"]["random_state"]
    workers = cfg["experiment"]["parallel_workers"]

    reference = _load_pipeline("p2", features_dir)
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

    reg_models = regressors()
    clf_models = classifiers()

    all_rows = []
    total_t = time.time()

    outer = tqdm(PIPELINES, desc="[run] pipelines", unit="pipe")
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
                            split_manifest=split_manifest)
            tqdm.write(f"[run] {pipe:>3} | {target:>15} | {name:<20} | "
                       f"{len(rows):3d} rows | {time.time()-t0:5.1f}s")
            return rows

        with tqdm_joblib(tqdm(total=len(reg_jobs),
                              desc=f"[run] {pipe} regressors", unit="fit")):
            res = Parallel(n_jobs=workers, backend="loky")(
                delayed(_job)(pipe, t, n, m, "regression")
                for t, (n, m) in reg_jobs
            )
        for r in res:
            all_rows.extend(r)

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
