"""DCIR extraction from the RPT pulse train, including unsupported protocols."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.dcir import (
    CANONICAL_C_RATE,
    canonical_dcir_per_file,
    identify_pulses,
    previous_rpt_dcir,
)

NOMINAL = 1.2


def _step(step_index: int, phase: str, duration: float, current: float,
          v_start: float, v_end: float, file_id: str = "f0") -> dict:
    return {
        "file_id": file_id,
        "step_index": step_index,
        "phase": phase,
        "duration_s": duration,
        "current_median_A": current,
        "voltage_start_V": v_start,
        "voltage_end_V": v_end,
        "test_type": "CU",
    }


def _pulse_train(file_id: str = "f0", resistance: float = 0.05,
                 ocv: float = 3.6) -> pd.DataFrame:
    """Long rest followed by a 12 s 0.5C discharge pulse, then a charge pulse."""
    current = -CANONICAL_C_RATE * NOMINAL          # -0.6 A
    rows = [
        _step(0, "rest", 7200.0, 0.0, ocv, ocv, file_id),
        _step(1, "discharge", 12.0, current, ocv, ocv + current * resistance, file_id),
        _step(2, "rest", 1800.0, 0.0, ocv, ocv, file_id),
        _step(3, "charge", 12.0, -current, ocv, ocv - current * resistance, file_id),
    ]
    return pd.DataFrame(rows)


def test_known_synthetic_current_step_recovers_the_resistance():
    steps = _pulse_train(resistance=0.05)
    out = canonical_dcir_per_file(steps, nominal_capacity_Ah=NOMINAL)
    assert len(out) == 1
    assert out["dcir_available"].iloc[0] == 1.0
    assert np.isclose(out["dcir_discharge_ohm"].iloc[0], 0.05, atol=1e-9)


def test_canonical_pulse_is_the_discharge_pulse():
    out = canonical_dcir_per_file(_pulse_train(), nominal_capacity_Ah=NOMINAL)
    assert out["dcir_pulse_current_A"].iloc[0] < 0.0
    assert np.isclose(out["dcir_pulse_duration_s"].iloc[0], 12.0)


def test_highest_soc_pulse_is_selected_for_reproducibility():
    """With several SOC levels the top-OCV pulse must win, not the first row."""
    low = _pulse_train(resistance=0.09, ocv=2.2)
    low["step_index"] = [10, 11, 12, 13]
    high = _pulse_train(resistance=0.05, ocv=3.6)
    high["step_index"] = [20, 21, 22, 23]
    out = canonical_dcir_per_file(
        pd.concat([low, high], ignore_index=True), nominal_capacity_Ah=NOMINAL,
    )
    assert np.isclose(out["dcir_pre_pulse_OCV_V"].iloc[0], 3.6)
    assert np.isclose(out["dcir_discharge_ohm"].iloc[0], 0.05, atol=1e-9)


def test_selection_is_independent_of_pulse_ordinal():
    """A missing early pulse must not shift which pulse is chosen."""
    full = _pulse_train(resistance=0.05, ocv=3.6)
    shifted = full.copy()
    shifted["step_index"] = shifted["step_index"] + 7
    a = canonical_dcir_per_file(full, nominal_capacity_Ah=NOMINAL)
    b = canonical_dcir_per_file(shifted, nominal_capacity_Ah=NOMINAL)
    assert np.isclose(
        a["dcir_discharge_ohm"].iloc[0], b["dcir_discharge_ohm"].iloc[0],
    )


def test_short_rest_disqualifies_the_pulse():
    """An unrelaxed cell gives a meaningless resistance, so reject the pulse."""
    steps = _pulse_train()
    steps.loc[0, "duration_s"] = 30.0          # far below the 600 s requirement
    assert identify_pulses(steps, nominal_capacity_Ah=NOMINAL).empty


def test_pulse_that_is_too_long_is_not_a_pulse():
    steps = _pulse_train()
    steps.loc[1, "duration_s"] = 3600.0
    pulses = identify_pulses(steps, nominal_capacity_Ah=NOMINAL)
    assert (pulses["phase"] != "discharge").all() or pulses.empty


def test_zero_current_step_is_rejected():
    steps = _pulse_train()
    steps.loc[1, "current_median_A"] = 0.0
    pulses = identify_pulses(steps, nominal_capacity_Ah=NOMINAL)
    assert "discharge" not in set(pulses["phase"])


def test_implausible_resistance_is_flagged_unavailable():
    steps = _pulse_train(resistance=50.0)      # far outside DCIR_BOUNDS_OHM
    out = canonical_dcir_per_file(steps, nominal_capacity_Ah=NOMINAL)
    assert out["dcir_available"].iloc[0] == 0.0
    assert np.isnan(out["dcir_discharge_ohm"].iloc[0])


def test_protocol_without_pulses_returns_empty_not_an_error():
    """Cycling blocks have no usable current step; that must not raise."""
    steps = pd.DataFrame([
        _step(0, "rest", 0.0, 0.0, 3.6, 3.6),
        _step(1, "discharge", 3600.0, -0.6, 3.6, 1.5),
        _step(2, "charge", 3600.0, 0.6, 1.5, 3.8),
    ])
    out = canonical_dcir_per_file(steps, nominal_capacity_Ah=NOMINAL)
    assert out.empty


def test_previous_rpt_dcir_joins_by_visit_and_computes_drift():
    steps = pd.concat([
        _pulse_train(file_id="v0", resistance=0.050),
        _pulse_train(file_id="v1", resistance=0.062),
    ], ignore_index=True)
    rpt = pd.DataFrame([
        {"condition": "c", "cell": "A", "visit": 0, "file_id": "v0"},
        {"condition": "c", "cell": "A", "visit": 1, "file_id": "v1"},
    ])
    out = previous_rpt_dcir(steps, rpt, nominal_capacity_Ah=NOMINAL)
    visit1 = out[out["visit"] == 1].iloc[0]
    assert np.isclose(visit1["dcir_discharge_ohm"], 0.062, atol=1e-9)
    assert np.isclose(visit1["dcir_delta_from_baseline_ohm"], 0.012, atol=1e-9)
    visit0 = out[out["visit"] == 0].iloc[0]
    assert np.isclose(visit0["dcir_delta_from_baseline_ohm"], 0.0, atol=1e-12)


def test_missing_pulse_yields_nan_and_zero_availability():
    steps = _pulse_train(file_id="v0")
    rpt = pd.DataFrame([
        {"condition": "c", "cell": "A", "visit": 0, "file_id": "v0"},
        {"condition": "c", "cell": "A", "visit": 1, "file_id": "absent"},
    ])
    out = previous_rpt_dcir(steps, rpt, nominal_capacity_Ah=NOMINAL)
    missing = out[out["visit"] == 1].iloc[0]
    assert missing["dcir_available"] == 0.0
    assert np.isnan(missing["dcir_discharge_ohm"])
