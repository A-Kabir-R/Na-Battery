import numpy as np
import pandas as pd
import pytest

from src.preprocessing.protocol_segmentation import (
    extract_reference_discharge,
    segment_cyc_protocol,
    summarize_steps,
)


def _protocol_frame(specification):
    rows = []
    time = 0.0
    ah = 0.0
    for step_id, current, duration, voltage_start, voltage_end in specification:
        samples = 8
        times = np.linspace(time, time + duration, samples)
        voltages = np.linspace(voltage_start, voltage_end, samples)
        for index, (sample_time, voltage) in enumerate(zip(times, voltages)):
            if index:
                ah += current * (duration / (samples - 1)) / 3600.0
            rows.append({
                "Time": sample_time,
                "Voltage": voltage,
                "Current": current,
                "Ah_counter": ah,
                "Temperature": 25.0 + 0.01 * len(rows),
                "StepID": step_id,
            })
        time += duration + 1.0
    return pd.DataFrame(rows)


def test_cycle_state_machine_keeps_boundaries_incomplete():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.0, 3.0),
        (2, 1.2, 300, 3.0, 3.8),
        (3, 0.0, 20, 3.8, 3.8),
        (4, -1.2, 300, 3.8, 3.0),
        (5, 0.0, 20, 3.0, 3.0),
        (6, 1.2, 300, 3.0, 3.8),
        (7, 0.0, 20, 3.8, 3.8),
        (8, -1.2, 300, 3.8, 3.0),
        (9, 0.0, 20, 3.0, 3.0),
        (10, 1.2, 300, 3.0, 3.8),
        (11, 0.0, 20, 3.8, 3.8),
        (12, -1.2, 300, 3.8, 3.0),
    ])
    steps = summarize_steps(frame)
    cycles = segment_cyc_protocol(
        frame,
        expected_charge_current_A=1.2,
        expected_discharge_current_A=1.2,
        steps=steps,
    )
    assert [cycle.cycle_in_file for cycle in cycles] == [0, 1, 2, 3]
    assert [cycle.complete for cycle in cycles] == [False, True, True, False]
    assert cycles[0].qc_flags == ("leading_unmatched_charge",)
    assert "missing_recharge" in cycles[-1].qc_flags


def test_reference_discharge_selects_full_range_05c_phase():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.0, 3.0),
        (2, -0.6, 1200, 3.0, 1.5),
        (3, 0.0, 20, 1.5, 1.5),
        (4, 0.6, 7200, 1.5, 3.8),
        (5, 0.0, 20, 3.8, 3.8),
        (6, -0.6, 7200, 3.8, 1.5),
        (7, 0.0, 20, 1.5, 1.5),
        (8, -0.06, 7200, 3.8, 1.5),
        (9, 0.0, 20, 1.5, 1.5),
    ])
    reference = extract_reference_discharge(frame)
    assert reference.status == "valid"
    assert reference.capacity_Ah == pytest.approx(1.2, rel=1e-3)


def test_reference_discharge_rejects_ambiguous_protocol():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.8, 3.8),
        (2, -0.6, 7200, 3.8, 1.5),
        (3, 0.0, 20, 1.5, 1.5),
        (4, 0.6, 7200, 1.5, 3.8),
        (5, 0.0, 20, 3.8, 3.8),
        (6, -0.6, 7200, 3.8, 1.5),
        (7, 0.0, 20, 1.5, 1.5),
    ])
    reference = extract_reference_discharge(frame)
    assert reference.status == "ambiguous"
    assert np.isnan(reference.capacity_Ah)


def test_narrow_diagnostic_pulse_is_retained_but_incomplete():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.60, 3.60),
        (2, 1.2, 12, 3.53, 3.60),
        (3, 0.0, 20, 3.60, 3.60),
        (4, -1.2, 12, 3.60, 3.53),
        (5, 0.0, 20, 3.53, 3.53),
        (6, 1.2, 12, 3.53, 3.60),
        (7, 0.0, 20, 3.60, 3.60),
        (8, -1.2, 12, 3.60, 3.53),
    ])
    cycles = segment_cyc_protocol(
        frame,
        expected_charge_current_A=1.2,
        expected_discharge_current_A=1.2,
        minimum_cycle_voltage_span_V=0.3,
    )
    paired = [cycle for cycle in cycles if cycle.charge_step_index is not None and cycle.cycle_in_file > 0]
    assert paired
    assert paired[0].complete is False
    assert "narrow_voltage_excursion" in paired[0].qc_flags


def test_low_capacity_full_voltage_cycle_remains_complete():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.80, 3.80),
        (2, -1.2, 120, 3.80, 1.50),
        (3, 0.0, 20, 1.50, 1.50),
        (4, 1.2, 120, 1.50, 3.80),
        (5, 0.0, 20, 3.80, 3.80),
        (6, -1.2, 120, 3.80, 1.50),
    ])
    cycles = segment_cyc_protocol(
        frame,
        expected_charge_current_A=1.2,
        expected_discharge_current_A=1.2,
        minimum_cycle_voltage_span_V=0.3,
    )
    complete = [cycle for cycle in cycles if cycle.complete]
    assert len(complete) == 1
    assert "narrow_voltage_excursion" not in complete[0].qc_flags


def test_full_discharge_with_partial_recharge_remains_incomplete():
    frame = _protocol_frame([
        (1, 0.0, 20, 3.80, 3.80),
        (2, -1.2, 120, 3.80, 1.50),
        (3, 0.0, 20, 1.50, 1.50),
        (4, 1.2, 120, 1.50, 1.65),
        (5, 0.0, 20, 1.65, 1.65),
        (6, -1.2, 120, 1.65, 1.50),
    ])
    cycles = segment_cyc_protocol(
        frame,
        expected_charge_current_A=1.2,
        expected_discharge_current_A=1.2,
        minimum_cycle_voltage_span_V=0.3,
        minimum_phase_voltage_span_V=0.25,
    )
    first = cycles[0]
    assert first.complete is False
    assert "narrow_charge_voltage_excursion" in first.qc_flags
