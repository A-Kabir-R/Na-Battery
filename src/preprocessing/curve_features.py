"""Voltage-resolved curve features: true Delta-Q(V), signed IC/DV peaks, thermal.

All functions here are pure and operate on plain arrays so they can be unit
tested against synthetic curves without touching any ``.ird`` file.

Causality
---------
Nothing in this module knows about RPTs or targets. The caller
(:mod:`src.preprocessing.unified`) is responsible for supplying only cycles that
precede the prediction anchor. The primary Delta-Q(V) reference is the first
complete cycle of the *same, already finished* cycling block, and the anchor is
by construction that block's last complete cycle — so both curves are in the
past at prediction time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

#: Epsilon guarding log() of a variance that can legitimately be exactly zero
#: (identical curves). log(0 + LOG_EPSILON) is a finite, reproducible floor.
LOG_EPSILON = 1.0e-12

#: Minimum samples in a phase before any curve feature is attempted.
MIN_CURVE_POINTS = 30

#: Minimum shared voltage interval for Delta-Q(V), in volts.
MIN_OVERLAP_SPAN_V = 0.10

#: Tolerance for non-monotonic accumulated capacity, in Ah.
MONOTONICITY_TOLERANCE_AH = 0.02

_DQV_NAN: dict[str, float] = {
    "dqv_logvar": np.nan,
    "dqv_min": np.nan,
    "dqv_mean": np.nan,
    "dqv_l1": np.nan,
    "dqv_l2": np.nan,
    "dqv_skew": np.nan,
    "dqv_kurtosis": np.nan,
    "dqv_v_at_max_abs_dev": np.nan,
    "dqv_curve_corr": np.nan,
    "dqv_overlap_span_V": np.nan,
    "dqv_valid_points": 0.0,
    "dqv_available": 0.0,
}

_PEAK_NAN: dict[str, float] = {
    "peak1_V": np.nan,
    "peak1_height": np.nan,
    "peak1_prominence": np.nan,
    "peak1_fwhm_V": np.nan,
    "peak2_V": np.nan,
    "peak2_height": np.nan,
    "peak_count": 0.0,
    "peaks_available": 0.0,
}


def _clean_monotone_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs, sort by x ascending, average duplicate x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return x, y
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    # Average duplicates without a pandas groupby round trip.
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size == x.size:
        return x, y
    sums = np.bincount(inverse, weights=y, minlength=unique_x.size)
    counts = np.bincount(inverse, minlength=unique_x.size)
    return unique_x, sums / np.maximum(counts, 1)


def discharge_qv_curve(voltage: np.ndarray, ah_counter: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (voltage_ascending, capacity_at_voltage) for one discharge phase.

    ``ah_counter`` is converted to capacity *removed since the start of the
    phase*, so the curve is comparable across cycles with different absolute
    counters.
    """
    voltage = np.asarray(voltage, dtype=float)
    ah = np.asarray(ah_counter, dtype=float)
    if voltage.size < MIN_CURVE_POINTS or ah.size != voltage.size:
        return np.empty(0), np.empty(0)
    finite = np.isfinite(ah)
    if not finite.any():
        return np.empty(0), np.empty(0)
    q = np.abs(ah - ah[finite][0])
    return _clean_monotone_xy(voltage, q)


def capacity_is_monotone(capacity: np.ndarray,
                         tolerance: float = MONOTONICITY_TOLERANCE_AH) -> bool:
    """True when accumulated capacity never decreases beyond ``tolerance``."""
    q = np.asarray(capacity, dtype=float)
    q = q[np.isfinite(q)]
    if q.size < 2:
        return False
    return bool(np.min(np.diff(q)) >= -abs(tolerance))


def delta_q_of_v(voltage_current: np.ndarray, capacity_current: np.ndarray,
                 voltage_reference: np.ndarray, capacity_reference: np.ndarray,
                 *, n_points: int = 100) -> dict[str, float]:
    """True voltage-resolved Delta-Q(V) = Q_current(V) - Q_reference(V).

    Both curves are interpolated onto a common grid spanning only their shared
    voltage interval; no extrapolation beyond measured coverage occurs. Returns
    ``_DQV_NAN`` (with ``dqv_available = 0``) whenever the curves cannot be
    compared.
    """
    v_cur, q_cur = _clean_monotone_xy(voltage_current, capacity_current)
    v_ref, q_ref = _clean_monotone_xy(voltage_reference, capacity_reference)
    if v_cur.size < 2 or v_ref.size < 2:
        return dict(_DQV_NAN)

    low = max(float(v_cur[0]), float(v_ref[0]))
    high = min(float(v_cur[-1]), float(v_ref[-1]))
    span = high - low
    if not np.isfinite(span) or span < MIN_OVERLAP_SPAN_V:
        out = dict(_DQV_NAN)
        out["dqv_overlap_span_V"] = float(max(span, 0.0)) if np.isfinite(span) else np.nan
        return out

    grid = np.linspace(low, high, int(n_points))
    q_cur_grid = np.interp(grid, v_cur, q_cur)
    q_ref_grid = np.interp(grid, v_ref, q_ref)
    delta = q_cur_grid - q_ref_grid
    finite = np.isfinite(delta)
    if finite.sum() < 3:
        out = dict(_DQV_NAN)
        out["dqv_overlap_span_V"] = float(span)
        return out
    delta = delta[finite]
    grid_valid = grid[finite]

    variance = float(np.var(delta))
    mean = float(np.mean(delta))
    std = float(np.std(delta))
    if std > 0:
        centered = (delta - mean) / std
        skew = float(np.mean(centered ** 3))
        kurtosis = float(np.mean(centered ** 4) - 3.0)
    else:
        # Identical curves: higher moments are undefined but the honest value
        # for a constant signal is zero shape deviation, not NaN.
        skew = 0.0
        kurtosis = 0.0

    if np.std(q_cur_grid[finite]) > 0 and np.std(q_ref_grid[finite]) > 0:
        corr = float(np.corrcoef(q_cur_grid[finite], q_ref_grid[finite])[0, 1])
    else:
        corr = np.nan

    return {
        "dqv_logvar": float(np.log(variance + LOG_EPSILON)),
        "dqv_min": float(np.min(delta)),
        "dqv_mean": mean,
        "dqv_l1": float(np.sum(np.abs(delta))),
        "dqv_l2": float(np.sqrt(np.sum(delta ** 2))),
        "dqv_skew": skew,
        "dqv_kurtosis": kurtosis,
        "dqv_v_at_max_abs_dev": float(grid_valid[int(np.argmax(np.abs(delta)))]),
        "dqv_curve_corr": corr,
        "dqv_overlap_span_V": float(span),
        "dqv_valid_points": float(finite.sum()),
        "dqv_available": 1.0,
    }


def signed_dqdv_peaks(voltage: np.ndarray, capacity: np.ndarray, *,
                      sigma: float = 3.0, n_peaks: int = 2) -> dict[str, float]:
    """First ``n_peaks`` reproducible dQ/dV extrema, ordered by voltage.

    Peak *location* is found on ``|dQ/dV|``, which is required here: the curve is
    parameterised as capacity-removed against ascending voltage, so for a
    discharge phase Q(V) decreases and every incremental-capacity peak is a
    minimum of the signed derivative. Searching signed maxima alone would return
    near-zero plateau points instead of the phase-transition peaks.

    Two things distinguish this from the legacy extractor, and they are what the
    revision actually required:

    * the reported height keeps the **sign** of dQ/dV, so charge and discharge
      peaks are not conflated;
    * peaks are ordered by **voltage position**, not by magnitude, so "peak 1"
      denotes the same physical transition in every fold. The legacy version
      reported only the single largest-magnitude peak, which could jump between
      transitions as a cell aged.
    """
    v, q = _clean_monotone_xy(voltage, capacity)
    out = dict(_PEAK_NAN)
    if v.size < MIN_CURVE_POINTS or float(v[-1] - v[0]) < 1e-3:
        return out
    if not capacity_is_monotone(q):
        return out

    q_smooth = gaussian_filter1d(q, sigma=float(sigma))
    dqdv = np.gradient(q_smooth, v)
    if not np.isfinite(dqdv).any():
        return out

    magnitude = np.abs(dqdv[np.isfinite(dqdv)])
    prominence = max(float(np.nanpercentile(magnitude, 90)) * 0.1, 1e-6)

    # Locate extrema on the magnitude curve, then read the SIGNED height back.
    found, properties = find_peaks(np.abs(dqdv), prominence=prominence)
    if found.size == 0:
        return out
    prominences = properties.get("prominences", np.zeros(found.size))
    ordered = sorted(
        ((int(index), float(prom)) for index, prom in zip(found, prominences)),
        key=lambda item: float(v[item[0]]),
    )

    out["peak_count"] = float(len(ordered))
    out["peaks_available"] = 1.0
    sample_index = np.arange(v.size, dtype=float)
    for rank, (index, prom) in enumerate(ordered[:n_peaks], start=1):
        out[f"peak{rank}_V"] = float(v[index])
        out[f"peak{rank}_height"] = float(dqdv[index])  # signed
        if rank == 1:
            out["peak1_prominence"] = float(prom)
            try:
                _, _, left, right = peak_widths(
                    np.abs(dqdv), [index], rel_height=0.5,
                )
                left_v = float(np.interp(left[0], sample_index, v))
                right_v = float(np.interp(right[0], sample_index, v))
                out["peak1_fwhm_V"] = abs(right_v - left_v)
            except Exception:
                out["peak1_fwhm_V"] = np.nan
    return out


def icdv_reference_shift(current: dict[str, float], reference: dict[str, float]
                         ) -> dict[str, float]:
    """Shift of the first dQ/dV peak between an anchor and a reference cycle."""
    available = (
        float(current.get("peaks_available", 0.0)) > 0.0
        and float(reference.get("peaks_available", 0.0)) > 0.0
        and np.isfinite(current.get("peak1_V", np.nan))
        and np.isfinite(reference.get("peak1_V", np.nan))
    )
    if not available:
        return {
            "icdv_dis_peak1_V_shift": np.nan,
            "icdv_dis_peak1_height_change": np.nan,
            "icdv_available": 0.0,
        }
    return {
        "icdv_dis_peak1_V_shift": float(current["peak1_V"] - reference["peak1_V"]),
        "icdv_dis_peak1_height_change": float(
            current["peak1_height"] - reference["peak1_height"]
        ),
        "icdv_available": 1.0,
    }


def thermal_features(*, discharge_time: np.ndarray, discharge_temperature: np.ndarray,
                     discharge_capacity: np.ndarray,
                     charge_time: np.ndarray, charge_temperature: np.ndarray
                     ) -> dict[str, float]:
    """Compact thermal descriptors for one cycle.

    Returns NaN for every field when the temperature channel is unusable, so a
    missing sensor is never silently imputed as zero rise.
    """
    nan_out = {
        "thermal_discharge_rise_degC": np.nan,
        "thermal_charge_rise_degC": np.nan,
        "thermal_max_dTdt_degC_s": np.nan,
        "thermal_mean_dTdt_degC_s": np.nan,
        "thermal_rise_per_Ah": np.nan,
        "thermal_asymmetry_degC": np.nan,
    }
    t_dis = np.asarray(discharge_temperature, dtype=float)
    time_dis = np.asarray(discharge_time, dtype=float)
    finite = np.isfinite(t_dis) & np.isfinite(time_dis)
    if finite.sum() < 3:
        return nan_out
    t_dis, time_dis = t_dis[finite], time_dis[finite]

    discharge_rise = float(t_dis[-1] - t_dis[0])
    duration = float(time_dis[-1] - time_dis[0])
    if duration > 0:
        dtdt = np.gradient(t_dis, time_dis)
        dtdt = dtdt[np.isfinite(dtdt)]
        max_dtdt = float(np.max(np.abs(dtdt))) if dtdt.size else np.nan
        mean_dtdt = float(np.mean(dtdt)) if dtdt.size else np.nan
    else:
        max_dtdt = mean_dtdt = np.nan

    q = np.asarray(discharge_capacity, dtype=float)
    q = q[np.isfinite(q)]
    discharged = float(np.abs(q[-1] - q[0])) if q.size >= 2 else np.nan
    rise_per_ah = (
        discharge_rise / discharged
        if np.isfinite(discharged) and discharged > 1e-9 else np.nan
    )

    t_ch = np.asarray(charge_temperature, dtype=float)
    t_ch = t_ch[np.isfinite(t_ch)]
    charge_rise = float(t_ch[-1] - t_ch[0]) if t_ch.size >= 2 else np.nan
    asymmetry = (
        charge_rise - discharge_rise if np.isfinite(charge_rise) else np.nan
    )

    return {
        "thermal_discharge_rise_degC": discharge_rise,
        "thermal_charge_rise_degC": charge_rise,
        "thermal_max_dTdt_degC_s": max_dtdt,
        "thermal_mean_dTdt_degC_s": mean_dtdt,
        "thermal_rise_per_Ah": rise_per_ah,
        "thermal_asymmetry_degC": asymmetry,
    }


def safe_divide(numerator: Any, denominator: Any, *,
                epsilon: float = 1.0e-12) -> float:
    """Division that returns NaN — never inf — for invalid denominators."""
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < epsilon:
        return float("nan")
    return num / den
