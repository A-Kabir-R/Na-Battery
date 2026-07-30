"""Synthetic tests for true Delta-Q(V), signed IC/DV peaks and thermal features."""
from __future__ import annotations

import numpy as np

from src.preprocessing.curve_features import (
    LOG_EPSILON,
    delta_q_of_v,
    discharge_qv_curve,
    icdv_reference_shift,
    safe_divide,
    signed_dqdv_peaks,
    thermal_features,
)


def _linear_discharge(n: int = 300, capacity: float = 1.2):
    """Q(V) for an idealised discharge: capacity removed rises as V falls."""
    voltage = np.linspace(1.5, 3.8, n)
    q_removed = capacity * (3.8 - voltage) / (3.8 - 1.5)
    return voltage, q_removed


def _peaked_discharge(peak1: float = 2.2, peak2: float = 3.1,
                      scale: float = 1.2, n: int = 600):
    voltage = np.linspace(1.5, 3.8, n)
    bumps = (
        np.exp(-((voltage - peak1) / 0.08) ** 2)
        + 0.6 * np.exp(-((voltage - peak2) / 0.10) ** 2)
    )
    q_removed = np.cumsum(bumps[::-1])[::-1]
    return voltage, q_removed / q_removed[0] * scale


# ---------------------------------------------------------------------------
# Delta-Q(V)
# ---------------------------------------------------------------------------
def test_identical_curves_give_near_zero_delta_q():
    voltage, q = _linear_discharge()
    out = delta_q_of_v(voltage, q, voltage, q)
    assert out["dqv_available"] == 1.0
    assert abs(out["dqv_l2"]) < 1e-9
    assert abs(out["dqv_mean"]) < 1e-9
    assert abs(out["dqv_min"]) < 1e-9
    # log of a genuinely zero variance is floored by the epsilon guard, not inf.
    assert np.isfinite(out["dqv_logvar"])
    assert out["dqv_logvar"] == np.log(LOG_EPSILON)


def test_shifted_curve_gives_nonzero_delta_q():
    voltage, q = _linear_discharge()
    out = delta_q_of_v(voltage, q * 0.9, voltage, q)
    assert out["dqv_available"] == 1.0
    assert out["dqv_l2"] > 1e-3
    assert out["dqv_min"] < 0.0          # aged cell delivers less capacity
    assert out["dqv_logvar"] > np.log(LOG_EPSILON)


def test_delta_q_uses_only_the_shared_voltage_interval():
    voltage, q = _linear_discharge()
    # Current curve covers the low half, reference the high half.
    out = delta_q_of_v(voltage[:200], q[:200], voltage[100:], q[100:])
    assert out["dqv_available"] == 1.0
    low = max(voltage[0], voltage[100])
    high = min(voltage[199], voltage[-1])
    assert np.isclose(out["dqv_overlap_span_V"], high - low, atol=1e-6)


def test_non_overlapping_curves_return_missing_safely():
    _, q = _linear_discharge()
    out = delta_q_of_v(
        np.linspace(1.5, 2.0, 300), q,
        np.linspace(3.0, 3.8, 300), q,
    )
    assert out["dqv_available"] == 0.0
    assert np.isnan(out["dqv_logvar"])
    assert np.isnan(out["dqv_l2"])


def test_too_few_points_returns_missing():
    out = delta_q_of_v([1.5, 1.6], [0.0, 0.1], [1.5, 1.6], [0.0, 0.1])
    assert out["dqv_available"] == 0.0


def test_delta_q_never_extrapolates_beyond_measured_coverage():
    voltage, q = _linear_discharge()
    out = delta_q_of_v(voltage[:150], q[:150], voltage, q)
    assert out["dqv_overlap_span_V"] <= float(voltage[149] - voltage[0]) + 1e-9


# ---------------------------------------------------------------------------
# Incremental capacity / differential voltage
# ---------------------------------------------------------------------------
def test_signed_peaks_are_found_and_ordered_by_voltage():
    voltage, q = _peaked_discharge()
    out = signed_dqdv_peaks(voltage, q, sigma=2)
    assert out["peaks_available"] == 1.0
    assert np.isclose(out["peak1_V"], 2.2, atol=0.02)
    assert np.isclose(out["peak2_V"], 3.1, atol=0.02)
    assert out["peak1_V"] < out["peak2_V"]


def test_peak_height_keeps_its_sign_for_discharge():
    voltage, q = _peaked_discharge()
    out = signed_dqdv_peaks(voltage, q, sigma=2)
    # Capacity-removed against ascending voltage is decreasing, so dQ/dV < 0.
    assert out["peak1_height"] < 0.0
    assert out["peak1_prominence"] > 0.0


def test_fwhm_is_positive_and_finite():
    voltage, q = _peaked_discharge()
    out = signed_dqdv_peaks(voltage, q, sigma=2)
    assert np.isfinite(out["peak1_fwhm_V"])
    assert out["peak1_fwhm_V"] > 0.0


def test_peak_ordering_is_stable_under_ageing():
    """Peak 1 must denote the same transition before and after degradation."""
    fresh = signed_dqdv_peaks(*_peaked_discharge(2.2, 3.1, 1.2), sigma=2)
    aged = signed_dqdv_peaks(*_peaked_discharge(2.15, 3.05, 1.08), sigma=2)
    assert np.isclose(aged["peak1_V"] - fresh["peak1_V"], -0.05, atol=0.02)


def test_icdv_reference_shift_recovers_the_applied_shift():
    fresh = signed_dqdv_peaks(*_peaked_discharge(2.2, 3.1, 1.2), sigma=2)
    aged = signed_dqdv_peaks(*_peaked_discharge(2.15, 3.1, 1.2), sigma=2)
    shift = icdv_reference_shift(aged, fresh)
    assert shift["icdv_available"] == 1.0
    assert np.isclose(shift["icdv_dis_peak1_V_shift"], -0.05, atol=0.02)


def test_insufficient_data_is_handled_gracefully():
    voltage, q = _peaked_discharge()
    out = signed_dqdv_peaks(voltage[:10], q[:10], sigma=2)
    assert out["peaks_available"] == 0.0
    assert np.isnan(out["peak1_V"])


def test_icdv_shift_missing_when_either_side_has_no_peak():
    voltage, q = _peaked_discharge()
    good = signed_dqdv_peaks(voltage, q, sigma=2)
    bad = signed_dqdv_peaks(voltage[:10], q[:10], sigma=2)
    assert icdv_reference_shift(good, bad)["icdv_available"] == 0.0
    assert icdv_reference_shift(bad, good)["icdv_available"] == 0.0


def test_non_monotonic_capacity_is_rejected():
    voltage = np.linspace(1.5, 3.8, 300)
    noisy = np.cos(np.linspace(0, 40, 300))  # swings far beyond tolerance
    assert signed_dqdv_peaks(voltage, noisy)["peaks_available"] == 0.0


def test_discharge_qv_curve_is_referenced_to_phase_start():
    voltage = np.linspace(3.8, 1.5, 300)
    ah = np.linspace(10.0, 11.2, 300)   # absolute counter with an offset
    v_out, q_out = discharge_qv_curve(voltage, ah)
    assert q_out.min() >= 0.0
    assert np.isclose(q_out.max(), 1.2, atol=1e-6)
    assert np.all(np.diff(v_out) > 0)   # voltage returned ascending


# ---------------------------------------------------------------------------
# Thermal + helpers
# ---------------------------------------------------------------------------
def test_thermal_features_on_a_known_ramp():
    time = np.linspace(0.0, 100.0, 101)
    temperature = 25.0 + 0.05 * time          # +5 degC over the phase
    capacity = np.linspace(0.0, 1.0, 101)
    out = thermal_features(
        discharge_time=time, discharge_temperature=temperature,
        discharge_capacity=capacity,
        charge_time=time, charge_temperature=25.0 + 0.02 * time,
    )
    assert np.isclose(out["thermal_discharge_rise_degC"], 5.0, atol=1e-6)
    assert np.isclose(out["thermal_max_dTdt_degC_s"], 0.05, atol=1e-6)
    assert np.isclose(out["thermal_rise_per_Ah"], 5.0, atol=1e-6)
    assert np.isclose(out["thermal_asymmetry_degC"], 2.0 - 5.0, atol=1e-6)


def test_thermal_features_missing_when_temperature_channel_is_unusable():
    time = np.linspace(0.0, 100.0, 101)
    out = thermal_features(
        discharge_time=time,
        discharge_temperature=np.full(101, np.nan),
        discharge_capacity=np.linspace(0.0, 1.0, 101),
        charge_time=time, charge_temperature=np.full(101, np.nan),
    )
    # A dead sensor must not be reported as zero temperature rise.
    assert np.isnan(out["thermal_discharge_rise_degC"])


def test_safe_divide_returns_nan_not_inf():
    assert np.isnan(safe_divide(1.0, 0.0))
    assert np.isnan(safe_divide(1.0, np.nan))
    assert np.isnan(safe_divide(np.inf, 2.0))
    assert safe_divide(1.0, 4.0) == 0.25
