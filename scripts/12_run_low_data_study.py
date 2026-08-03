#!/usr/bin/env python3
"""Learning curves in units of training *cells*.

The data-efficiency claim the journal cares about is "physics constraints reduce
the aging-test burden", and the currency of that burden is cells, not anchors: a
cell is weeks of cycler time, an anchor is free once the cell exists.

For each outer fold the training pool is subsampled to 20/40/60/80/100% of its
cells, condition-stratified, repeated so the curve carries a band rather than a
single lucky draw. Test folds are held fixed across every budget so the curve
measures the budget and nothing else.

Writes, under ``artifacts/results/low_data/``:

    low_data_records.parquet   one row per (model, fold, fraction, repeat)
    low_data_curve.csv         median + IQR band per budget
    low_data_cells_to_reach.csv  cells each model needs to hit the reference error

Usage::

    python3 scripts/12_run_low_data_study.py
    python3 scripts/12_run_low_data_study.py --repeats 5 --models ConstantRate Ridge
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.low_data import (  # noqa: E402
    DEFAULT_FRACTIONS, LowDataStudy, cells_to_reach, summarize_curve,
)
from src.io.loaders import load_config  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402
from src.models.degradation_baselines import build_degradation_baselines  # noqa: E402
from src.models.regressors import STUDY_RIDGE_ALPHA  # noqa: E402
from src.splits.group_kfold import make_folds  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _statistical_models(feature_columns: list[str]):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def wrap(estimator, scale_target: bool = False):
        inner = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), estimator,
        )
        if not scale_target:
            return inner
        return TransformedTargetRegressor(
            regressor=inner, transformer=StandardScaler(),
        )

    return {
        # Frozen penalty, not RidgeCV: see STUDY_RIDGE_ALPHA. It matters more
        # here than anywhere else -- at the 20% budget a row-level LOO search
        # would tune alpha on a handful of cells' own anchors.
        "Ridge": wrap(Ridge(alpha=STUDY_RIDGE_ALPHA)),
        "ExtraTrees": wrap(ExtraTreesRegressor(n_estimators=400, random_state=42)),
        "GradientBoosting": wrap(GradientBoostingRegressor(random_state=42)),
        # The neural baseline needs a standardised target: the raw delta has a
        # scale of ~0.007 Ah and the optimiser stalls on it otherwise.
        "MLP": wrap(MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                 max_iter=8000, random_state=42), scale_target=True),
    }


def _load(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    artifacts = Path(cfg["paths"]["artifacts"])
    unified = artifacts / "features" / "unified.parquet"
    manifest = artifacts / "features" / "unified_feature_manifest.json"
    if not unified.exists():
        raise SystemExit(
            f"missing {unified}. Run scripts/01_build_features.py first, or point "
            f"SIB_ARTIFACTS at a built artifacts directory."
        )
    frame = pd.read_parquet(unified)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    return frame, json.loads(manifest.read_text())["feature_columns"]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20,
                        help="cell draws per budget (default 20)")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of model names; default is all")
    parser.add_argument("--reference-model", default="ExtraTrees",
                        help="model whose full-data error defines the threshold")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    frame, feature_columns = _load(cfg)
    out = args.output_dir or (
        Path(cfg["paths"]["artifacts"]) / "results" / "low_data"
    )
    out.mkdir(parents=True, exist_ok=True)

    models: dict[str, object] = {
        **build_degradation_baselines(),
        **_statistical_models(feature_columns),
    }
    if args.models:
        unknown = set(args.models) - set(models)
        if unknown:
            raise SystemExit(f"unknown model(s): {sorted(unknown)}")
        models = {name: models[name] for name in args.models}

    print(f"[low-data] anchors={len(frame)} cells={frame['cell'].nunique()} "
          f"models={len(models)} repeats={args.repeats}")

    records: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(
        make_folds(frame, n_splits=args.folds)
    ):
        pool = frame.iloc[train_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)
        print(f"[low-data] fold {fold}: pool {pool['cell'].nunique()} cells, "
              f"test {test['cell'].nunique()} cells")

        for name, template in models.items():
            def fit_predict(train, test_frame, _template=template, _name=name):
                model = copy.deepcopy(_template)
                if _name in build_degradation_baselines():
                    model.fit(train, train[TARGET].to_numpy(dtype=float))
                    return model.predict(test_frame)
                model.fit(train[feature_columns], train[TARGET])
                return model.predict(test_frame[feature_columns])

            study = LowDataStudy(
                fractions=tuple(args.fractions), repeats=args.repeats, seed=fold,
            )
            records.append(study.run(
                pool, test, TARGET, fit_predict, model_name=name, fold=fold,
            ))

    if not records:
        print("[low-data] nothing ran")
        return 1

    all_records = pd.concat(records, ignore_index=True)
    all_records.to_parquet(out / "low_data_records.parquet", index=False)
    failures = int((all_records["status"] != "ok").sum())
    if failures:
        print(f"[low-data] WARNING: {failures} failed fits (see records)")

    curve = summarize_curve(all_records)
    curve.to_csv(out / "low_data_curve.csv", index=False)

    # How many cells each model needs to match the reference model's full-data
    # error. This is the sentence the paper reduces to: "the proposed model
    # reaches the same accuracy with N fewer cells."
    reference = curve[
        curve["model"].eq(args.reference_model) & (curve["fraction"] >= 1.0)
    ]
    rows = []
    if reference.empty:
        print(f"[low-data] reference model {args.reference_model} not in curve")
    else:
        threshold = float(reference["median"].iloc[0])
        for name in curve["model"].unique():
            rows.append({
                "model": name,
                "threshold_cell_macro_MAE": threshold,
                "reference_model": args.reference_model,
                "cells_to_reach": cells_to_reach(
                    curve[curve["model"].eq(name)], threshold,
                ),
            })
        pd.DataFrame(rows).to_csv(out / "low_data_cells_to_reach.csv", index=False)
        print(f"\n[low-data] threshold = {threshold:.5f} Ah "
              f"({args.reference_model} at full data)")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + curve.to_string(index=False))
    print(f"\n[low-data] wrote {out}")
    return 0


def main() -> int:
    """Run under a capture log so a finished run is still debuggable.

    Everything these studies printed used to go to stdout and nowhere else.
    """
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "low_data" / "logs"
    with run_log("12_run_low_data_study", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
