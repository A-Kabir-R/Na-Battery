#!/usr/bin/env python3
"""Render every manuscript figure that the study scripts produce.

The existing figure scripts (04, 08) cover training diagnostics and parity
plots. The four study scripts wrote CSVs that nothing rendered, so none of the
paper's actual claims had a figure. This closes that.

Reads only frozen results tables — never a model — so it runs on CPU in seconds
and a figure cannot disagree with the table it came from. Every input is
optional: a missing study is reported and skipped, not fatal, so this is usable
part-way through a run.

Writes to ``artifacts/results/paper_figures/`` in every format configured under
``plotting.formats`` (vector PDF + raster PNG by default).

Usage::

    python3 scripts/16_make_paper_figures.py
    python3 scripts/16_make_paper_figures.py --only condition_matrix calibration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from src.evaluation import paper_figures as pf  # noqa: E402
from src.io.loaders import load_config  # noqa: E402
from src.pinn.logging_utils import run_log  # noqa: E402

TARGET = "delta_next_rpt_Q_Ah"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as error:  # noqa: BLE001
        print(f"[fig] could not read {path}: {error}")
        return None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None,
                        help="render only these figures by name")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    results = artifacts / "results"
    out = args.output_dir or (results / "paper_figures")
    out.mkdir(parents=True, exist_ok=True)

    plotting = cfg.get("plotting") or {}
    style = {
        "formats": tuple(str(f).lstrip(".") for f in
                         (plotting.get("formats") or ("pdf", "png"))),
        "dpi": int(plotting.get("dpi", 300)),
    }
    print(f"[fig] output={out}  formats={style['formats']}  dpi={style['dpi']}")

    anchors = _read(artifacts / "features" / "unified.parquet")
    if anchors is not None and TARGET in anchors.columns:
        anchors = anchors[anchors[TARGET].notna()].reset_index(drop=True)

    generalization = _read(results / "generalization" / "generalization_metrics.csv")
    low_data = _read(results / "low_data" / "low_data_curve.csv")
    coverage = _read(results / "uncertainty" / "coverage_by_condition.csv")
    intervals = _read(results / "uncertainty" / "uncertainty_predictions.parquet")
    robustness = _read(results / "robustness" / "robustness_summary.csv")
    components = _read(results / "rate_components.csv")
    surface = _read(results / "response_surface.csv")

    # Persistence floor for the learning-curve reference line: predicting no
    # change at all, which is exactly mean(|delta|).
    persistence = None
    if anchors is not None and TARGET in anchors.columns:
        persistence = float(anchors[TARGET].abs().mean())

    rendered: list[str] = []
    skipped: list[str] = []

    def run(name: str, function, *fargs, **fkwargs) -> None:
        if args.only and name not in args.only:
            return
        try:
            function(*fargs, **fkwargs, **style)
            rendered.append(name)
        except Exception as error:  # noqa: BLE001
            # One bad table must not abort the whole figure set; a half-written
            # figure directory is worse than a reported gap.
            print(f"[fig] FAILED {name}: {type(error).__name__}: {error}")
            skipped.append(f"{name} (error)")

    if anchors is not None:
        run("condition_matrix", pf.fig_condition_matrix, anchors, out)
        run("signal_to_noise_regimes", pf.fig_signal_to_noise_regimes,
            anchors, out, target=TARGET)
    else:
        skipped.append("dataset figures (no unified.parquet)")

    if generalization is not None:
        for study in generalization["study"].dropna().unique():
            run(f"generalization_{study}", pf.fig_generalization,
                generalization, out, study=str(study))
        # Paired bootstrap straight off the per-split predictions.
        predictions = _read(
            results / "generalization" / "generalization_predictions.parquet"
        )
        if predictions is not None and {"model", "cell", "y_true", "y_pred"} <= set(
            predictions.columns
        ):
            per_cell = predictions.assign(
                abs_error=(predictions["y_true"] - predictions["y_pred"]).abs()
            )[["model", "cell", "abs_error"]]
            run("paired_comparison", pf.fig_paired_comparison, per_cell, out,
                reference="ZeroChange")
    else:
        skipped.append("generalization (run scripts/11)")

    if low_data is not None:
        run("low_data_learning_curve", pf.fig_low_data_curve, low_data, out,
            persistence_mae=persistence)
    else:
        skipped.append("low-data curve (run scripts/12)")

    if coverage is not None:
        alpha = 0.1
        if intervals is not None and "alpha" in intervals.columns:
            alpha = float(intervals["alpha"].dropna().iloc[0])
        run("calibration_by_condition", pf.fig_calibration, coverage, out,
            alpha=alpha)
    else:
        skipped.append("calibration (run scripts/13)")

    if intervals is not None:
        run("prediction_intervals", pf.fig_prediction_intervals, intervals, out)

    if robustness is not None:
        run("robustness", pf.fig_robustness, robustness, out)
    else:
        skipped.append("robustness (run scripts/14)")

    # Physics figures depend on a hybrid-rate run having exported its
    # components; they are optional until that model is trained.
    if components is not None:
        run("rate_decomposition", pf.fig_rate_decomposition, components, out)
    else:
        skipped.append("rate decomposition (needs a HybridRateModel export)")
    if surface is not None:
        run("response_surface", pf.fig_response_surface, surface, out,
            x="T_degC", y="DOD_pct")
    else:
        skipped.append("response surface (needs a HybridRateModel export)")

    print(f"\n[fig] rendered {len(rendered)}: {', '.join(rendered) or '(none)'}")
    if skipped:
        print(f"[fig] skipped {len(skipped)}:")
        for item in skipped:
            print(f"        - {item}")
    return 0 if rendered else 1


def main() -> int:
    log_dir = Path(load_config()["paths"]["artifacts"]) / "results" / "paper_figures" / "logs"
    with run_log("16_make_paper_figures", log_dir, argv=sys.argv[1:]):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
