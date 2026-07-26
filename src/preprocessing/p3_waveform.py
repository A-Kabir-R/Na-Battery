"""P3 waveform features using the shared validated cycle segments."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from tqdm.auto import tqdm

from ..io.ird_parser import parse_ird
from ..io.loaders import load_config, load_cycle_metrics, load_inventory
from .protocol_segmentation import segment_cyc_protocol, summarize_steps

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    grouped = pd.DataFrame({"x": x[order], "y": y[order]}).groupby("x", sort=True)["y"].mean()
    return grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)


def _resample_vq(voltage: np.ndarray, capacity: np.ndarray,
                 n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Resample V(Q) and return the physical Ah grid with voltage."""
    q, v = _unique_xy(capacity, voltage)
    if len(q) < 2 or q[-1] - q[0] < 1e-9:
        return np.full(n, np.nan), np.full(n, np.nan)
    grid = np.linspace(q[0], q[-1], n)
    return grid, np.interp(grid, q, v)


def _dqdv_peaks(voltage: np.ndarray, capacity: np.ndarray,
                sigma: float) -> Dict[str, float]:
    v, q = _unique_xy(voltage, capacity)
    if len(v) < 30 or v[-1] - v[0] < 1e-3:
        return {
            "dqdv_peak_count": 0.0,
            "dqdv_peak_h": np.nan,
            "dqdv_peak_V": np.nan,
            "dqdv_peak_fwhm_V": np.nan,
        }
    q_smooth = gaussian_filter1d(q, sigma=sigma)
    dqdv = np.gradient(q_smooth, v)
    finite = np.isfinite(dqdv)
    if not finite.any():
        return {
            "dqdv_peak_count": 0.0,
            "dqdv_peak_h": np.nan,
            "dqdv_peak_V": np.nan,
            "dqdv_peak_fwhm_V": np.nan,
        }
    prominence = max(float(np.nanpercentile(np.abs(dqdv[finite]), 90)) * 0.1, 1e-6)
    peaks, _ = find_peaks(np.abs(dqdv), prominence=prominence)
    if len(peaks) == 0:
        return {
            "dqdv_peak_count": 0.0,
            "dqdv_peak_h": np.nan,
            "dqdv_peak_V": np.nan,
            "dqdv_peak_fwhm_V": np.nan,
        }
    primary = int(peaks[np.argmax(np.abs(dqdv[peaks]))])
    _, _, left_ips, right_ips = peak_widths(np.abs(dqdv), [primary], rel_height=0.5)
    sample_index = np.arange(len(v), dtype=float)
    left_voltage = float(np.interp(left_ips[0], sample_index, v))
    right_voltage = float(np.interp(right_ips[0], sample_index, v))
    return {
        "dqdv_peak_count": float(len(peaks)),
        "dqdv_peak_h": float(dqdv[primary]),
        "dqdv_peak_V": float(v[primary]),
        "dqdv_peak_fwhm_V": abs(right_voltage - left_voltage),
    }


def _phase_features(part: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    time = part["Time"].to_numpy(dtype=float)
    temperature = part["Temperature"].to_numpy(dtype=float)
    return {
        f"wf_{prefix}_duration_s": float(time[-1] - time[0]),
        f"wf_{prefix}_current_std_A": float(part["Current"].std()),
        f"wf_{prefix}_temperature_rise_degC": float(temperature[-1] - temperature[0]),
        f"wf_{prefix}_temperature_slope_degC_s": (
            float(np.polyfit(time, temperature, 1)[0]) if len(time) > 2 else np.nan
        ),
    }


def _waveform_cycle_features(discharge: pd.DataFrame, charge: pd.DataFrame,
                             sigma: float, n_resample: int) -> Dict[str, float]:
    q_dis = discharge["Ah_counter"].to_numpy(dtype=float)
    q_dis = np.abs(q_dis - q_dis[0])
    v_dis = discharge["Voltage"].to_numpy(dtype=float)
    q_grid, voltage_grid = _resample_vq(v_dis, q_dis, n=n_resample)
    valid = np.isfinite(q_grid) & np.isfinite(voltage_grid)
    if valid.sum() < 3:
        area = slope = curvature = np.nan
    else:
        area = float(_trapezoid(voltage_grid[valid], q_grid[valid]))
        span = float(q_grid[valid][-1] - q_grid[valid][0])
        slope = float((voltage_grid[valid][-1] - voltage_grid[valid][0]) / span) if span else np.nan
        middle = voltage_grid[valid][valid.sum() // 2]
        curvature = float(middle - 0.5 * (voltage_grid[valid][0] + voltage_grid[valid][-1]))
    peak_features = _dqdv_peaks(v_dis, q_dis, sigma=sigma)
    return {
        "wf_vq_area_VAh": area,
        "wf_vq_mean_slope_V_Ah": slope,
        "wf_vq_mid_curvature_V": curvature,
        **_phase_features(discharge, prefix="discharge"),
        **_phase_features(charge, prefix="charge"),
        **{f"wf_{key}": value for key, value in peak_features.items()},
    }


def _process_file(row: Dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    protocol = cfg["protocol"]
    path = Path(row["path"])
    result: dict[str, Any] = {
        "file_id": row.get("file_id"),
        "path": str(path),
        "rows": [],
        "status": "failed",
        "error": None,
    }
    try:
        parsed = parse_ird(
            path,
            strict=True,
            max_invalid_fraction=float(protocol["max_invalid_fraction"]),
            max_sampling_gap_s=float(protocol["max_sampling_gap_s"]),
        )
        assert parsed.data is not None
        data = parsed.data
        data = data.copy()
        for column, limits in protocol["physical_bounds"].items():
            invalid = (data[column] < limits[0]) | (data[column] > limits[1])
            data.loc[invalid, column] = np.nan
        steps = summarize_steps(data, current_deadband_A=float(protocol["current_deadband_A"]))
        nominal = float(protocol["nominal_capacity_Ah"])
        segments = segment_cyc_protocol(
            data,
            expected_charge_current_A=nominal * float(row["C_ch"]),
            expected_discharge_current_A=nominal * float(row["C_dis"]),
            current_relative_tolerance=float(protocol["cycling_current_relative_tolerance"]),
            minimum_phase_capacity_Ah=float(protocol["minimum_phase_capacity_Ah"]),
            capacity_ratio_bounds=tuple(protocol["capacity_ratio_bounds"]),
            minimum_cycle_voltage_span_V=float(
                protocol.get("minimum_cycle_voltage_span_V", 0.0)
            ),
            minimum_phase_voltage_span_V=float(
                protocol.get("minimum_phase_voltage_span_V", 0.0)
            ),
            steps=steps,
        )
        rows: list[dict[str, Any]] = []
        for segment in segments:
            if not segment.complete or segment.discharge_step_index is None or segment.charge_step_index is None:
                continue
            discharge_step = steps[segment.discharge_step_index]
            charge_step = steps[segment.charge_step_index]
            features = _waveform_cycle_features(
                data.iloc[discharge_step.start:discharge_step.stop],
                data.iloc[charge_step.start:charge_step.stop],
                sigma=float(cfg["preprocessing"]["dqdv_smooth_sigma"]),
                n_resample=int(cfg["preprocessing"]["vq_resample_points"]),
            )
            features.update({
                "condition": row["condition"],
                "cell": row["cell"],
                "visit": int(row["visit"]),
                "file_id": row.get("file_id"),
                "cycle_in_file": segment.cycle_in_file,
                "path": str(path),
            })
            rows.append(features)
        if not rows:
            raise ValueError("no complete waveform cycles extracted")
        result.update({"rows": rows, "status": "valid"})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_p3(workers: int | None = None, limit_files: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    workers = workers or int(cfg["experiment"]["parallel_workers"])
    inventory = load_inventory()
    inventory = inventory[
        (inventory["test_type"] == "CYC") &
        (inventory.get("parse_status", "valid") == "valid")
    ].reset_index(drop=True)
    if limit_files:
        inventory = inventory.head(limit_files)
    canonical_cycles = load_cycle_metrics()
    canonical_cycles = canonical_cycles[canonical_cycles["cycle_complete"].fillna(False)]
    selected_file_ids = set(inventory["file_id"].dropna())
    expected_keys = set(
        canonical_cycles.loc[
            canonical_cycles["file_id"].isin(selected_file_ids),
            ["file_id", "cycle_in_file"],
        ].itertuples(index=False, name=None)
    )
    expected_by_file: dict[str, set[tuple[Any, Any]]] = {}
    for key in expected_keys:
        expected_by_file.setdefault(key[0], set()).add(key)
    tasks = inventory.to_dict(orient="records")
    all_rows: List[Dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_process_file, tasks, chunksize=1)
        bar = tqdm(iterator, total=len(tasks), desc="[p3] waveform files", unit="file")
        for result in bar:
            all_rows.extend(result["rows"])
            extracted_keys = {
                (row["file_id"], row["cycle_in_file"]) for row in result["rows"]
            }
            expected_file_keys = expected_by_file.get(result["file_id"], set())
            missing_count = len(expected_file_keys - extracted_keys)
            extra_count = len(extracted_keys - expected_file_keys)
            coverage_status = result["status"]
            if missing_count or extra_count:
                coverage_status = "cycle_key_mismatch"
            coverage.append({
                "file_id": result["file_id"],
                "path": result["path"],
                "status": coverage_status,
                "expected_cycle_count": len(expected_file_keys),
                "cycle_count": len(result["rows"]),
                "missing_cycle_count": missing_count,
                "extra_cycle_count": extra_count,
                "error": result["error"],
            })
            bar.set_postfix_str(f"cycles={len(all_rows)} failures={sum(r['status'] != 'valid' for r in coverage)}")
    coverage_frame = pd.DataFrame(coverage)
    coverage_path = Path(cfg["paths"]["artifacts"]) / "qc" / "p3_coverage.parquet"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_parquet(coverage_path, index=False)
    coverage_fraction = float((coverage_frame["status"] == "valid").mean()) if len(coverage_frame) else 0.0
    minimum = float(cfg["preprocessing"]["minimum_p3_file_coverage"])
    if coverage_fraction < minimum:
        raise RuntimeError(
            f"P3 file coverage {coverage_fraction:.3f} is below required {minimum:.3f}; "
            f"inspect {coverage_path}"
        )
    actual_keys = {(row["file_id"], row["cycle_in_file"]) for row in all_rows}
    cycle_coverage = len(actual_keys & expected_keys) / len(expected_keys) if expected_keys else 0.0
    minimum_cycle_coverage = float(cfg["preprocessing"]["minimum_p3_cycle_coverage"])
    if cycle_coverage < minimum_cycle_coverage or actual_keys - expected_keys:
        raise RuntimeError(
            f"P3 cycle-key coverage {cycle_coverage:.3f} is below required "
            f"{minimum_cycle_coverage:.3f} or contains unexpected keys; inspect {coverage_path}"
        )
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    frame = build_p3(limit_files=3)
    print("P3 sample shape:", frame.shape)
    print(frame.head())
