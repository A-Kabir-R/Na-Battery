"""Perturbations must corrupt inputs without touching the truth column."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.robustness import (
    FAMILY_FLAGS,
    FEATURE_FAMILIES,
    build_perturbations,
    drop_feature_family,
    run_robustness,
)


def _frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "T_degC": np.full(n, 25.0),
        "T_max": rng.normal(30, 1, n),
        "temperature_span_C": rng.normal(5, 0.5, n),
        "DOD_pct": np.full(n, 100.0),
        "C_ch": np.full(n, 1.0),
        "C_dis": np.full(n, 1.0),
        "Q_discharge_Ah": rng.normal(1.1, 0.01, n),
        "EFC_cum": np.arange(n, dtype=float) * 40,
        "cumulative_Ah_throughput": np.arange(n, dtype=float) * 48,
        "previous_rpt_Q_Ah": rng.normal(1.15, 0.01, n),
        "voltage_hysteresis_V": rng.normal(0.1, 0.01, n),
        "target": rng.normal(-0.007, 0.002, n),
    })
    for family, columns in FEATURE_FAMILIES.items():
        for column in columns:
            frame[column] = rng.normal(0, 1, n)
        flag = FAMILY_FLAGS.get(family)
        if flag:
            frame[flag] = 1.0
    return frame


def test_perturbations_never_alter_the_target():
    frame = _frame()
    rng = np.random.default_rng(0)
    for perturbation in build_perturbations():
        out = perturbation(frame, rng)
        assert np.allclose(out["target"], frame["target"]), (
            f"{perturbation.name} modified the truth column; the study would "
            "then measure a changed problem, not a degraded input"
        )


def test_perturbations_do_not_mutate_the_input_frame():
    frame = _frame()
    before = frame.copy()
    rng = np.random.default_rng(0)
    for perturbation in build_perturbations():
        perturbation(frame, rng)
    pd.testing.assert_frame_equal(frame, before)


def test_dropping_a_family_clears_its_availability_flag():
    frame = _frame()
    rng = np.random.default_rng(0)
    out = drop_feature_family("resistance")(frame, rng)
    for column in FEATURE_FAMILIES["resistance"]:
        assert out[column].isna().all()
    assert (out[FAMILY_FLAGS["resistance"]] == 0.0).all(), (
        "a masked family must also report itself unavailable, or it arrives as "
        "a silently imputed median"
    )


def test_temperature_offset_shifts_temperature_only():
    frame = _frame()
    out = build_perturbations(temperature_offset_C=3.0)[0](
        frame, np.random.default_rng(0),
    )
    assert np.allclose(out["T_degC"], frame["T_degC"] + 3.0)
    assert np.allclose(out["DOD_pct"], frame["DOD_pct"])


def test_report_includes_an_unperturbed_reference_row():
    frame = _frame()
    report = run_robustness(
        frame, "target", lambda f: np.zeros(len(f)),
        perturbations=build_perturbations()[:3],
    )
    assert (report["perturbation"] == "none").sum() == 1
    reference = report[report["perturbation"] == "none"].iloc[0]
    assert reference["MAE_ratio"] == 1.0


def test_degradation_shows_up_as_a_ratio_above_one():
    frame = _frame()
    slope = -2.0e-4

    def predict(f: pd.DataFrame) -> np.ndarray:
        # A model that leans entirely on temperature.
        return slope * (f["T_degC"].to_numpy() - 20.0)

    # Make the model exactly right on the clean inputs, so a thermal bias can
    # only hurt. With an arbitrary target the offset can accidentally move the
    # prediction *toward* the truth, which measures the fixture, not robustness.
    frame["target"] = predict(frame) + np.random.default_rng(0).normal(
        0.0, 1e-6, len(frame),
    )

    report = run_robustness(
        frame, "target", predict,
        perturbations=[build_perturbations(temperature_offset_C=50.0)[0]],
    )
    offset = report[report["perturbation"] == "temperature_offset"].iloc[0]
    assert offset["MAE_ratio"] > 1.0
    # A 50 C bias through a 2e-4/C slope is a 0.01 shift, far above the 1e-6
    # noise floor, so the degradation should be orders of magnitude.
    assert offset["MAE"] > 100 * report[
        report["perturbation"] == "none"
    ].iloc[0]["MAE"]


def test_interval_widening_is_reported_when_available():
    frame = _frame()
    report = run_robustness(
        frame, "target", lambda f: np.zeros(len(f)),
        perturbations=[drop_feature_family("resistance")],
        interval_half_width=lambda f: np.where(
            f["dcir_available"].to_numpy() > 0, 0.01, 0.05,
        ),
    )
    row = report[report["perturbation"] == "missing_resistance"].iloc[0]
    assert row["width_ratio"] > 1.0
    assert bool(row["widened"]) is True, (
        "a model that loses a diagnostic and stays equally confident is the "
        "failure mode this study exists to catch"
    )


def test_failed_perturbation_is_recorded_not_swallowed():
    frame = _frame()

    def explode(f: pd.DataFrame) -> np.ndarray:
        if "dcir_prev_rpt_discharge_ohm" in f and f[
            "dcir_prev_rpt_discharge_ohm"
        ].isna().all():
            raise RuntimeError("model cannot handle missing DCIR")
        return np.zeros(len(f))

    report = run_robustness(
        frame, "target", explode, perturbations=[drop_feature_family("resistance")],
    )
    row = report[report["perturbation"] == "missing_resistance"].iloc[0]
    assert row["status"] == "failed"
    assert "cannot handle" in row["error"]
