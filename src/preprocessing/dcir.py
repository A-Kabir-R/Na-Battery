"""DC internal resistance from the RPT/CU pulse train.

Protocol findings (inspected from ``canonical/steps.parquet``, 289 CU files)
---------------------------------------------------------------------------
Cycling (CYC) blocks contain no usable current step for resistance: the median
in-block rest duration is 0 s. Resistance is therefore measured *only* in
RPT/CU files, which carry a highly regular pulse train:

    3 SOC levels (pre-pulse rest OCV ~3.60 / ~2.95 / ~2.20 V)
      x 3 C-rates (0.5C, 1C, 2C)
      x 2 directions (discharge, charge)
    = 18 pulses of 12 s each, every pulse preceded by a >= 1800 s rest
      (7200 s before the 0.5C discharge pulse at each SOC level).

285 of 289 CU files contain the full train. Three files have a gap that shifts
the pulse ordinal, so pulses are selected by *physical* identity
(phase + C-rate + pre-pulse OCV rank), never by position in the file.

Canonical measurement
---------------------
    R_dc = (V_pulse_end - V_rest_end) / (I_pulse - 0)

taken from the **0.5C discharge pulse at the highest pre-pulse OCV** of the
previous RPT. This is a 12 s response. Sub-second and 1 s responses would
require re-parsing the raw ``.ird`` waveforms; the 12 s value is available
directly from the canonical step table and is consistent across every file, so
it is the only resistance response registered.

Causality
---------
Only the *previous* RPT is used. The next RPT defines the target and is never
touched. Cells whose previous RPT lacks the pulse (or is the initial visit with
no predecessor) receive NaN and ``dcir_available = 0``.

Relaxation features are deliberately not implemented: although the 1800 s and
7200 s post-pulse rests could in principle support an exponential fit, that was
scoped out of this revision. The protocol does support it and it remains open
work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Target C-rate of the canonical pulse.
CANONICAL_C_RATE = 0.5

#: Tolerance on the identified C-rate.
C_RATE_TOLERANCE = 0.15

#: Pulse duration window, seconds.
PULSE_DURATION_RANGE_S = (5.0, 60.0)

#: Minimum pre-pulse rest for the cell to count as relaxed.
MINIMUM_PRE_PULSE_REST_S = 600.0

#: Minimum pulse current magnitude, amperes.
MINIMUM_PULSE_CURRENT_A = 0.1

#: Physically plausible DCIR window for a 1.2 Ah sodium-ion cell, ohms.
DCIR_BOUNDS_OHM = (0.005, 1.0)

_NAN_ROW = {
    "dcir_discharge_ohm": np.nan,
    "dcir_pulse_current_A": np.nan,
    "dcir_pulse_duration_s": np.nan,
    "dcir_pre_pulse_OCV_V": np.nan,
    "dcir_available": 0.0,
}


def identify_pulses(steps: pd.DataFrame, *, nominal_capacity_Ah: float) -> pd.DataFrame:
    """Return one row per valid current pulse found in ``steps``.

    ``steps`` must be a canonical step table for RPT/CU files, carrying
    ``file_id``, ``step_index``, ``phase``, ``duration_s``, ``current_median_A``,
    ``voltage_start_V`` and ``voltage_end_V``.
    """
    required = {
        "file_id", "step_index", "phase", "duration_s",
        "current_median_A", "voltage_end_V",
    }
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"steps table is missing columns: {sorted(missing)}")

    frame = steps.sort_values(["file_id", "step_index"]).copy()
    grouped = frame.groupby("file_id", sort=False)
    frame["previous_phase"] = grouped["phase"].shift(1)
    frame["previous_duration_s"] = grouped["duration_s"].shift(1)
    frame["previous_voltage_end_V"] = grouped["voltage_end_V"].shift(1)

    low, high = PULSE_DURATION_RANGE_S
    is_pulse = (
        frame["phase"].isin(["discharge", "charge"])
        & frame["duration_s"].between(low, high)
        & frame["current_median_A"].abs().gt(MINIMUM_PULSE_CURRENT_A)
        & frame["previous_phase"].eq("rest")
        & frame["previous_duration_s"].ge(MINIMUM_PRE_PULSE_REST_S)
    )
    pulses = frame[is_pulse].copy()
    if pulses.empty:
        return pulses

    pulses["C_rate"] = (
        pulses["current_median_A"].abs() / float(nominal_capacity_Ah)
    )
    pulses["pre_pulse_OCV_V"] = pulses["previous_voltage_end_V"]
    # R = dV / dI, with the pre-pulse rest current taken as exactly zero.
    pulses["dcir_ohm"] = (
        (pulses["voltage_end_V"] - pulses["previous_voltage_end_V"])
        / pulses["current_median_A"]
    )
    return pulses


def canonical_dcir_per_file(steps: pd.DataFrame, *, nominal_capacity_Ah: float
                            ) -> pd.DataFrame:
    """One canonical DCIR measurement per RPT/CU file.

    Selection is by physical identity: the discharge pulse whose C-rate is
    closest to :data:`CANONICAL_C_RATE` at the *highest* pre-pulse OCV. Ties are
    broken by the lowest ``step_index`` so the result is deterministic.
    """
    pulses = identify_pulses(steps, nominal_capacity_Ah=nominal_capacity_Ah)
    columns = ["file_id", *_NAN_ROW.keys()]
    if pulses.empty:
        return pd.DataFrame(columns=columns)

    candidates = pulses[
        pulses["phase"].eq("discharge")
        & (pulses["C_rate"] - CANONICAL_C_RATE).abs().le(C_RATE_TOLERANCE)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    # Highest pre-pulse OCV == top-of-charge pulse, the most reproducible point.
    candidates = candidates.sort_values(
        ["file_id", "pre_pulse_OCV_V", "step_index"],
        ascending=[True, False, True],
    )
    selected = candidates.groupby("file_id", as_index=False).first()

    lower, upper = DCIR_BOUNDS_OHM
    plausible = selected["dcir_ohm"].between(lower, upper)
    out = pd.DataFrame({
        "file_id": selected["file_id"].to_numpy(),
        "dcir_discharge_ohm": np.where(
            plausible, selected["dcir_ohm"].to_numpy(), np.nan
        ),
        "dcir_pulse_current_A": selected["current_median_A"].to_numpy(),
        "dcir_pulse_duration_s": selected["duration_s"].to_numpy(),
        "dcir_pre_pulse_OCV_V": selected["pre_pulse_OCV_V"].to_numpy(),
        "dcir_available": plausible.to_numpy().astype(float),
    })
    return out[columns]


def previous_rpt_dcir(steps: pd.DataFrame, rpt_measurements: pd.DataFrame, *,
                      nominal_capacity_Ah: float,
                      initial_visit: int = 0) -> pd.DataFrame:
    """Per-(condition, cell, visit) DCIR of the *previous* RPT and its drift.

    Returns one row per RPT visit carrying that visit's own DCIR plus the
    deviation from the cell's first valid RPT DCIR. The caller joins this onto
    anchors using the anchor's ``previous_rpt_visit``, so no future RPT is read.
    """
    per_file = canonical_dcir_per_file(
        steps, nominal_capacity_Ah=nominal_capacity_Ah
    )
    keys = ["condition", "cell", "visit", "file_id"]
    missing = [key for key in keys if key not in rpt_measurements.columns]
    if missing:
        raise ValueError(f"rpt_measurements is missing columns: {missing}")

    visits = rpt_measurements[keys].drop_duplicates("file_id")
    merged = visits.merge(per_file, on="file_id", how="left")
    merged["dcir_available"] = merged["dcir_available"].fillna(0.0)

    baseline = (
        merged[merged["visit"].eq(initial_visit) & merged["dcir_available"].gt(0)]
        .sort_values(["condition", "cell", "visit"])
        .groupby(["condition", "cell"])["dcir_discharge_ohm"]
        .first()
        .rename("dcir_baseline_ohm")
    )
    merged = merged.join(baseline, on=["condition", "cell"])
    merged["dcir_delta_from_baseline_ohm"] = (
        merged["dcir_discharge_ohm"] - merged["dcir_baseline_ohm"]
    )
    return merged.sort_values(["condition", "cell", "visit"]).reset_index(drop=True)
