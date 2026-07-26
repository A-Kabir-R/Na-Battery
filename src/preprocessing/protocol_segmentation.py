"""Protocol-aware step, cycle, and reference-discharge extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

PhaseKind = Literal["charge", "discharge", "rest", "transition"]


@dataclass(frozen=True)
class StepSegment:
    step_index: int
    step_id: int
    start: int
    stop: int
    phase: PhaseKind
    duration_s: float
    current_median_A: float
    current_abs_p90_A: float
    capacity_Ah: float
    energy_Wh: float
    voltage_start_V: float
    voltage_end_V: float
    voltage_min_V: float
    voltage_max_V: float
    temperature_min_degC: float
    temperature_max_degC: float
    bounded_start: bool
    bounded_end: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CycleSegment:
    cycle_in_file: int
    start: int
    stop: int
    discharge_step_index: int | None
    charge_step_index: int | None
    complete: bool
    qc_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["qc_flags"] = ";".join(self.qc_flags)
        return row


@dataclass(frozen=True)
class ReferenceDischarge:
    step_index: int | None
    capacity_Ah: float
    current_A: float
    start_time_s: float
    end_time_s: float
    status: str
    qc_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["qc_flags"] = ";".join(self.qc_flags)
        return row


def _integral(t: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    t_valid = t[valid]
    y_valid = y[valid]
    if np.any(np.diff(t_valid) < 0):
        return float("nan")
    return float(_trapezoid(y_valid, t_valid))


def _finite_stat(values: np.ndarray, function) -> float:
    finite = values[np.isfinite(values)]
    return float(function(finite)) if finite.size else float("nan")


def infer_charge_current_sign(df: pd.DataFrame, current_deadband_A: float = 0.02) -> int:
    """Infer which current sign charges the cell from voltage and Ah trends."""
    votes: list[float] = []
    step_ids = df["StepID"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(step_ids[1:] != step_ids[:-1]) + 1, len(df)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        part = df.iloc[start:stop]
        current = float(part["Current"].median())
        if abs(current) <= current_deadband_A or len(part) < 2:
            continue
        dv = float(part["Voltage"].iloc[-1] - part["Voltage"].iloc[0])
        dah = float(part["Ah_counter"].iloc[-1] - part["Ah_counter"].iloc[0])
        trend = dv if abs(dv) >= 0.01 else dah
        if abs(trend) >= 1e-6:
            votes.append(np.sign(current) * np.sign(trend) * abs(dah))
    score = float(np.nansum(votes))
    if abs(score) < 1e-9:
        raise ValueError("could not infer charge-current sign from voltage/Ah trends")
    return 1 if score > 0 else -1


def summarize_steps(df: pd.DataFrame, *, current_deadband_A: float = 0.02,
                    charge_current_sign: int | None = None) -> list[StepSegment]:
    """Summarize every contiguous StepID run without bridging intervening gaps."""
    if df.empty:
        return []
    sign = charge_current_sign or infer_charge_current_sign(df, current_deadband_A)
    step_ids = df["StepID"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(step_ids[1:] != step_ids[:-1]) + 1, len(df)]
    steps: list[StepSegment] = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        part = df.iloc[start:stop]
        t = part["Time"].to_numpy(dtype=float)
        voltage = part["Voltage"].to_numpy(dtype=float)
        current = part["Current"].to_numpy(dtype=float)
        temperature = part["Temperature"].to_numpy(dtype=float)
        current_median = float(np.nanmedian(current))
        current_abs_p90 = float(np.nanpercentile(np.abs(current), 90))
        if current_abs_p90 <= current_deadband_A:
            phase: PhaseKind = "rest"
        elif np.sign(current_median) == sign:
            phase = "charge"
        elif np.sign(current_median) == -sign:
            phase = "discharge"
        else:
            phase = "transition"
        capacity = abs(_integral(t, current)) / 3600.0
        energy = abs(_integral(t, voltage * current)) / 3600.0
        steps.append(StepSegment(
            step_index=index,
            step_id=int(step_ids[start]),
            start=int(start),
            stop=int(stop),
            phase=phase,
            duration_s=float(t[-1] - t[0]) if len(t) else float("nan"),
            current_median_A=current_median,
            current_abs_p90_A=current_abs_p90,
            capacity_Ah=capacity,
            energy_Wh=energy,
            voltage_start_V=float(voltage[0]),
            voltage_end_V=float(voltage[-1]),
            voltage_min_V=_finite_stat(voltage, np.min),
            voltage_max_V=_finite_stat(voltage, np.max),
            temperature_min_degC=_finite_stat(temperature, np.min),
            temperature_max_degC=_finite_stat(temperature, np.max),
            bounded_start=start > 0,
            bounded_end=stop < len(df),
        ))
    return steps


def _at_expected_rate(step: StepSegment, expected_current_A: float,
                      relative_tolerance: float) -> bool:
    low = expected_current_A * (1.0 - relative_tolerance)
    high = expected_current_A * (1.0 + relative_tolerance)
    return low <= step.current_abs_p90_A <= high


def segment_cyc_protocol(df: pd.DataFrame, *, expected_charge_current_A: float,
                         expected_discharge_current_A: float,
                         current_relative_tolerance: float = 0.35,
                         minimum_phase_capacity_Ah: float = 0.001,
                         capacity_ratio_bounds: tuple[float, float] = (0.25, 4.0),
                         minimum_cycle_voltage_span_V: float = 0.0,
                         minimum_phase_voltage_span_V: float = 0.0,
                         steps: Sequence[StepSegment] | None = None) -> list[CycleSegment]:
    """Pair expected-rate discharge steps with their following recharge steps.

    Protocol preambles at another C-rate are ignored. A leading expected-rate
    charge and a trailing discharge are retained as incomplete boundary cycles.
    """
    phase_steps = list(steps or summarize_steps(df))
    discharge = [
        step for step in phase_steps
        if step.phase == "discharge"
        and step.capacity_Ah >= minimum_phase_capacity_Ah
        and _at_expected_rate(step, expected_discharge_current_A, current_relative_tolerance)
    ]
    # Recharge can become CV-dominated after severe degradation, so its p90
    # current may fall below the nominal programmed rate. Pair by order and
    # capacity balance after identifying cycles from expected-rate discharges.
    charge = [
        step for step in phase_steps
        if step.phase == "charge"
        and step.capacity_Ah >= minimum_phase_capacity_Ah
    ]
    if not discharge and not charge:
        return []

    segments: list[CycleSegment] = []
    discharge_positions = {step.step_index: step for step in discharge}
    charge_positions = {step.step_index: step for step in charge}
    used_charge: set[int] = set()
    cycle_number = 0

    first_discharge_index = min(discharge_positions, default=len(phase_steps))
    leading_charges = [
        step for step in charge
        if step.step_index < first_discharge_index
        and _at_expected_rate(step, expected_charge_current_A, current_relative_tolerance)
    ]
    for step in leading_charges:
        used_charge.add(step.step_index)
        segments.append(CycleSegment(
            cycle_in_file=0,
            start=step.start,
            stop=step.stop,
            discharge_step_index=None,
            charge_step_index=step.step_index,
            complete=False,
            qc_flags=("leading_unmatched_charge",),
        ))

    for discharge_step in discharge:
        cycle_number += 1
        next_discharge = min(
            (step.step_index for step in discharge if step.step_index > discharge_step.step_index),
            default=len(phase_steps),
        )
        all_candidates = [
            step for step in charge
            if discharge_step.step_index < step.step_index < next_discharge
            and step.step_index not in used_charge
        ]
        candidates = [
            step for step in all_candidates
            if capacity_ratio_bounds[0]
            <= discharge_step.capacity_Ah / step.capacity_Ah
            <= capacity_ratio_bounds[1]
        ]
        charge_step = candidates[0] if candidates else None
        flags: list[str] = []
        if not discharge_step.bounded_start or not discharge_step.bounded_end:
            flags.append("discharge_touches_file_boundary")
        if charge_step is None:
            flags.append("missing_recharge")
        else:
            used_charge.add(charge_step.step_index)
            if not charge_step.bounded_start or not charge_step.bounded_end:
                flags.append("charge_touches_file_boundary")
            ratio = discharge_step.capacity_Ah / charge_step.capacity_Ah
            if not capacity_ratio_bounds[0] <= ratio <= capacity_ratio_bounds[1]:
                flags.append("capacity_ratio_implausible")
            discharge_voltage_span = (
                discharge_step.voltage_max_V - discharge_step.voltage_min_V
            )
            charge_voltage_span = charge_step.voltage_max_V - charge_step.voltage_min_V
            if (
                np.isfinite(discharge_voltage_span)
                and discharge_voltage_span < minimum_phase_voltage_span_V
            ):
                flags.append("narrow_discharge_voltage_excursion")
            if (
                np.isfinite(charge_voltage_span)
                and charge_voltage_span < minimum_phase_voltage_span_V
            ):
                flags.append("narrow_charge_voltage_excursion")
            voltage_min = min(discharge_step.voltage_min_V, charge_step.voltage_min_V)
            voltage_max = max(discharge_step.voltage_max_V, charge_step.voltage_max_V)
            if (
                np.isfinite(voltage_min)
                and np.isfinite(voltage_max)
                and voltage_max - voltage_min < minimum_cycle_voltage_span_V
            ):
                flags.append("narrow_voltage_excursion")
        complete = not flags
        segments.append(CycleSegment(
            cycle_in_file=cycle_number,
            start=discharge_step.start,
            stop=charge_step.stop if charge_step else discharge_step.stop,
            discharge_step_index=discharge_step.step_index,
            charge_step_index=charge_step.step_index if charge_step else None,
            complete=complete,
            qc_flags=tuple(flags),
        ))
    return segments


def extract_reference_discharge(df: pd.DataFrame, *, reference_current_A: float = 0.6,
                                current_relative_tolerance: float = 0.2,
                                charged_voltage_min_V: float = 3.6,
                                discharged_voltage_max_V: float = 1.8,
                                capacity_bounds_Ah: tuple[float, float] = (0.2, 2.0),
                                steps: Sequence[StepSegment] | None = None) -> ReferenceDischarge:
    """Extract the unique full-range 0.5C discharge from a CU/RPT protocol."""
    phase_steps = list(steps or summarize_steps(df))
    candidates = [
        step for step in phase_steps
        if step.phase == "discharge"
        and _at_expected_rate(step, reference_current_A, current_relative_tolerance)
        and step.voltage_start_V >= charged_voltage_min_V
        and step.voltage_end_V <= discharged_voltage_max_V
        and capacity_bounds_Ah[0] <= step.capacity_Ah <= capacity_bounds_Ah[1]
        and step.bounded_start
        and step.bounded_end
    ]
    if not candidates:
        return ReferenceDischarge(None, np.nan, np.nan, np.nan, np.nan,
                                  "missing", ("no_reference_discharge",))
    if len(candidates) > 1:
        return ReferenceDischarge(None, np.nan, np.nan, np.nan, np.nan,
                                  "ambiguous", ("multiple_reference_discharges",))
    step = candidates[0]
    time = df["Time"].to_numpy(dtype=float)
    return ReferenceDischarge(
        step_index=step.step_index,
        capacity_Ah=step.capacity_Ah,
        current_A=step.current_abs_p90_A,
        start_time_s=float(time[step.start]),
        end_time_s=float(time[step.stop - 1]),
        status="valid",
        qc_flags=(),
    )


def aggregate_cycle(df: pd.DataFrame, steps: Sequence[StepSegment],
                    segment: CycleSegment) -> dict[str, Any]:
    """Aggregate one segment without using samples outside its phase bounds."""
    discharge = steps[segment.discharge_step_index] if segment.discharge_step_index is not None else None
    charge = steps[segment.charge_step_index] if segment.charge_step_index is not None else None
    part = df.iloc[segment.start:segment.stop]
    q_dis = discharge.capacity_Ah if discharge else np.nan
    q_charge = charge.capacity_Ah if charge else np.nan
    e_dis = discharge.energy_Wh if discharge else np.nan
    e_charge = charge.energy_Wh if charge else np.nan
    return {
        "cycle_in_file": segment.cycle_in_file,
        "cycle_complete": segment.complete,
        "cycle_qc_status": "valid" if segment.complete else "incomplete",
        "cycle_qc_flags": ";".join(segment.qc_flags),
        "start_row": segment.start,
        "stop_row": segment.stop,
        "cycle_start_time_s": float(part["Time"].iloc[0]),
        "cycle_end_time_s": float(part["Time"].iloc[-1]),
        "Q_charge_Ah": q_charge,
        "Q_discharge_Ah": q_dis,
        "E_charge_Wh": e_charge,
        "E_discharge_Wh": e_dis,
        "coulombic_efficiency": q_dis / q_charge if q_charge and np.isfinite(q_charge) else np.nan,
        "energy_efficiency": e_dis / e_charge if e_charge and np.isfinite(e_charge) else np.nan,
        "V_max": float(part["Voltage"].max()),
        "V_min": float(part["Voltage"].min()),
        "T_max": float(part["Temperature"].max()),
        "T_min": float(part["Temperature"].min()),
        "T_mean": float(part["Temperature"].mean()),
        "Ah_span": float(part["Ah_counter"].max() - part["Ah_counter"].min()),
        "duration_s": float(part["Time"].iloc[-1] - part["Time"].iloc[0]),
    }
