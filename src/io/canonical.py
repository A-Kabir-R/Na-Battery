"""Build versioned canonical tables directly from Standard-cycling IRD files."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .ird_parser import FILENAME_RE, parse_condition, parse_ird
from .loaders import load_config
from ..preprocessing.protocol_segmentation import (
    aggregate_cycle,
    extract_reference_discharge,
    segment_cyc_protocol,
    summarize_steps,
)

MANIFEST_COLUMNS = [
    "file_id", "relative_path", "path", "checksum_sha256", "size_bytes", "mtime_ns",
    "condition", "cell", "visit", "test_id", "filename_test_type", "test_type",
    "rpt_T_degC", "DOD_pct", "C_ch", "C_dis", "T_degC", "start_time",
    "test_name", "row_count", "parse_status", "parse_error_code",
    "qc_status",
]


@dataclass(frozen=True)
class CanonicalBuildResult:
    manifest: pd.DataFrame
    steps: pd.DataFrame
    cycles: pd.DataFrame
    rpt_measurements: pd.DataFrame
    qc_flags: pd.DataFrame


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_standard_cycling_files(dataset_root: Path | str) -> list[Path]:
    """Discover only the two permitted Standard-cycling raw-file roles."""
    root = Path(dataset_root).resolve()
    files: set[Path] = set()
    condition_dirs = list(root.glob("DOD*_*"))
    if not condition_dirs:
        raise ValueError(
            f"dataset_root must directly contain Standard-cycling DOD condition folders: {root}"
        )
    for condition_dir in condition_dirs:
        condition_fields = parse_condition(condition_dir.name)
        if not condition_dir.is_dir() or pd.isna(condition_fields["DOD_pct"]):
            continue
        cycling = condition_dir / "Cycling_Periods"
        if cycling.is_dir():
            files.update(path.resolve() for path in cycling.glob("*.ird"))
        for cu_dir in condition_dir.glob("CU_at_*degC"):
            if cu_dir.is_dir():
                files.update(path.resolve() for path in cu_dir.glob("*.ird"))
    for path in files:
        if root not in path.parents:
            raise ValueError(f"discovered path outside configured Standard-cycling scope: {path}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _identity(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    condition = path.parents[1].name
    role = "CYC" if path.parent.name == "Cycling_Periods" else "CU"
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"filename does not match the IRD naming contract: {path.name}")
    parsed = match.groupdict()
    condition_fields = parse_condition(condition)
    return {
        "file_id": sha256(relative.encode("utf-8")).hexdigest()[:20],
        "relative_path": relative,
        "path": str(path),
        "condition": condition,
        "cell": parsed["cell"],
        "visit": int(parsed["visit"]),
        "test_id": int(parsed["testid"]),
        "filename_test_type": parsed["test"],
        "test_type": role,
        "rpt_T_degC": int(parsed["rpt_T"]),
        **condition_fields,
    }


def _qc_rows(file_id: str, issues: Iterable[Any], *, level: str = "file") -> list[dict[str, Any]]:
    rows = []
    for issue in issues:
        row = issue.to_dict()
        row.update({"file_id": file_id, "level": level, "action": "review"})
        rows.append(row)
    return rows


def _signal_qc(file_id: str, path: Path, data: pd.DataFrame,
               bounds: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, limits in bounds.items():
        if column not in data.columns:
            continue
        low, high = float(limits[0]), float(limits[1])
        invalid = (data[column] < low) | (data[column] > high)
        count = int(invalid.sum())
        if count:
            rows.append({
                "file_id": file_id,
                "level": "sample",
                "path": str(path),
                "stage": "signal_qc",
                "code": f"{column.lower()}_out_of_range",
                "message": f"{count}/{len(data)} {column} values outside [{low}, {high}]",
                "severity": "error",
                "row": None,
                "column": column,
                "value": None,
                "exception_type": None,
                "action": "exclude_invalid_samples_and_review_file",
            })
    return rows


def _clean_physical_bounds(data: pd.DataFrame,
                           bounds: dict[str, list[float]]) -> pd.DataFrame:
    cleaned = data.copy()
    for column, limits in bounds.items():
        invalid = (cleaned[column] < limits[0]) | (cleaned[column] > limits[1])
        cleaned.loc[invalid, column] = np.nan
    return cleaned


def _write_samples(data: pd.DataFrame, identity: dict[str, Any], output_dir: Path,
                   bounds: dict[str, list[float]]) -> None:
    sample_dir = (
        output_dir / "samples" / f"condition={identity['condition']}" /
        f"cell={identity['cell']}"
    )
    samples = data.copy()
    for column in bounds:
        samples[f"raw_{column}"] = samples[column]
    samples = _clean_physical_bounds(samples, bounds)
    samples.insert(0, "row_in_file", np.arange(len(samples), dtype=np.int64))
    samples.insert(0, "file_id", identity["file_id"])
    _atomic_parquet(samples, sample_dir / f"{identity['file_id']}.parquet")


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def build_canonical_tables(*, dataset_root: Path | str | None = None,
                           output_dir: Path | str | None = None,
                           limit_files: int | None = None,
                           write_samples: bool = False,
                           checksum_files: bool = True) -> CanonicalBuildResult:
    """Parse raw files and atomically write canonical manifests and aggregates."""
    cfg = load_config()
    root = Path(dataset_root or cfg["paths"]["dataset_root"]).resolve()
    output = Path(output_dir or cfg["paths"]["canonical"]).resolve()
    files = discover_standard_cycling_files(root)
    if limit_files is not None:
        files = files[:limit_files]

    manifest_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    rpt_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    protocol = cfg["protocol"]

    for path in tqdm(files, desc="[canonical] parse IRD", unit="file"):
        try:
            identity = _identity(path, root)
        except Exception as exc:
            relative = path.relative_to(root).as_posix()
            file_id = sha256(relative.encode("utf-8")).hexdigest()[:20]
            qc_rows.append({
                "file_id": file_id, "level": "file", "path": str(path), "stage": "identity",
                "code": "identity_error", "message": str(exc), "severity": "error",
                "row": None, "column": None, "value": None,
                "exception_type": type(exc).__name__, "action": "exclude_and_review",
            })
            continue

        stat = path.stat()
        parsed = parse_ird(
            path,
            strict=False,
            max_invalid_fraction=float(protocol["max_invalid_fraction"]),
            max_sampling_gap_s=float(protocol["max_sampling_gap_s"]),
        )
        checksum = _checksum(path) if checksum_files else None
        start_time = parsed.metadata.get("starttime") if parsed.metadata else pd.NaT
        manifest_row = {
            **identity,
            "checksum_sha256": checksum,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "start_time": start_time,
            "test_name": parsed.metadata.get("testname") if parsed.metadata else None,
            "row_count": int(parsed.qc.get("row_count", 0)),
            "parse_status": "valid" if parsed.qc.get("parse_ok") else "failed",
            "parse_error_code": parsed.qc.get("parse_error_code"),
            "qc_status": "pending",
        }
        manifest_rows.append(manifest_row)
        qc_rows.extend(_qc_rows(identity["file_id"], parsed.issues))
        if identity["filename_test_type"] != identity["test_type"]:
            qc_rows.append({
                "file_id": identity["file_id"], "level": "file", "path": str(path),
                "stage": "identity", "code": "filename_role_mismatch",
                "message": f"filename says {identity['filename_test_type']} but directory role is {identity['test_type']}",
                "severity": "warning", "row": None, "column": None, "value": None,
                "exception_type": None, "action": "retain_directory_role_and_review",
            })
        if parsed.data is None or not parsed.qc.get("parse_ok"):
            continue

        raw_data = parsed.data
        signal_issues = _signal_qc(identity["file_id"], path, raw_data, protocol["physical_bounds"])
        qc_rows.extend(signal_issues)
        if write_samples:
            _write_samples(raw_data, identity, output, protocol["physical_bounds"])
        data = _clean_physical_bounds(raw_data, protocol["physical_bounds"])

        try:
            steps = summarize_steps(
                data,
                current_deadband_A=float(protocol["current_deadband_A"]),
            )
        except Exception as exc:
            qc_rows.append({
                "file_id": identity["file_id"], "level": "file", "path": str(path),
                "stage": "segmentation", "code": "charge_sign_inference_failed",
                "message": str(exc), "severity": "error", "row": None, "column": "Current",
                "value": None, "exception_type": type(exc).__name__, "action": "exclude_and_review",
            })
            continue
        for step in steps:
            step_rows.append({**identity, **step.to_dict()})

        if identity["test_type"] == "CYC":
            nominal = float(protocol["nominal_capacity_Ah"])
            segments = segment_cyc_protocol(
                data,
                expected_charge_current_A=nominal * float(identity["C_ch"]),
                expected_discharge_current_A=nominal * float(identity["C_dis"]),
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
            if not segments:
                qc_rows.append({
                    "file_id": identity["file_id"], "level": "file", "path": str(path),
                    "stage": "segmentation", "code": "no_cycles_detected",
                    "message": "no expected-rate cycling phases were detected", "severity": "error",
                    "row": None, "column": None, "value": None, "exception_type": None,
                    "action": "exclude_and_review",
                })
            for segment_index, segment in enumerate(segments):
                cycle_rows.append({
                    **identity,
                    "file_start_time": start_time,
                    "segment_index": segment_index,
                    **aggregate_cycle(data, steps, segment),
                })
        else:
            reference = extract_reference_discharge(
                data,
                reference_current_A=float(protocol["rpt_reference_current_A"]),
                current_relative_tolerance=float(protocol["rpt_current_relative_tolerance"]),
                charged_voltage_min_V=float(protocol["rpt_charged_voltage_min_V"]),
                discharged_voltage_max_V=float(protocol["rpt_discharged_voltage_max_V"]),
                capacity_bounds_Ah=tuple(protocol["rpt_capacity_bounds_Ah"]),
                steps=steps,
            )
            rpt_rows.append({
                **identity,
                "file_start_time": start_time,
                "measurement_type": "reference_discharge_0.5C",
                "reference_capacity_Ah": reference.capacity_Ah,
                "reference_current_A": reference.current_A,
                "measurement_start_s": reference.start_time_s,
                "measurement_end_s": reference.end_time_s,
                "rpt_qc_status": reference.status,
                "rpt_qc_flags": ";".join(reference.qc_flags),
                "selection_rule": "unique 0.5C high-voltage-to-low-voltage discharge",
            })

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    steps_df = pd.DataFrame(step_rows)
    cycles = pd.DataFrame(cycle_rows)
    rpt = pd.DataFrame(rpt_rows)
    qc = pd.DataFrame(qc_rows)

    if not manifest.empty:
        if manifest["file_id"].duplicated().any() or manifest["relative_path"].duplicated().any():
            raise AssertionError("file manifest primary keys are not unique")
    if not steps_df.empty and steps_df.duplicated(["file_id", "step_index"]).any():
        raise AssertionError("step primary keys are not unique")
    if not cycles.empty and cycles.duplicated(["file_id", "segment_index"]).any():
        raise AssertionError("cycle segment primary keys are not unique")
    if not cycles.empty:
        complete_keys = cycles[cycles["cycle_complete"]]
        if complete_keys.duplicated(["file_id", "cycle_in_file"]).any():
            raise AssertionError("complete cycle merge keys are not unique")
    if not rpt.empty and rpt["file_id"].duplicated().any():
        raise AssertionError("RPT file keys are not unique")

    if not cycles.empty:
        cycles = cycles.sort_values(
            ["condition", "cell", "file_start_time", "test_id", "cycle_in_file"],
            na_position="last",
        ).reset_index(drop=True)
        cycles["global_cycle"] = pd.Series(pd.NA, index=cycles.index, dtype="Int64")
        complete = cycles["cycle_complete"].fillna(False).astype(bool)
        cycles.loc[complete, "global_cycle"] = (
            cycles.loc[complete].groupby(["condition", "cell"]).cumcount() + 1
        ).astype("Int64")
        cycles["cumulative_Ah_throughput"] = np.nan
        cycles.loc[complete, "cumulative_Ah_throughput"] = (
            cycles.loc[complete].groupby(["condition", "cell"])["Q_discharge_Ah"].cumsum()
        )

    if not rpt.empty:
        valid = rpt[rpt["rpt_qc_status"] == "valid"]
        duplicate_counts = valid.groupby(["condition", "cell", "visit"]).size()
        for key, count in duplicate_counts[duplicate_counts > 1].items():
            cond, cell, visit = key
            qc_rows_for_key = valid[
                (valid["condition"] == cond) & (valid["cell"] == cell) & (valid["visit"] == visit)
            ]
            timestamps = pd.to_datetime(qc_rows_for_key["file_start_time"], errors="coerce")
            distinguishable = timestamps.notna().all() and not timestamps.duplicated().any()
            for _, row in qc_rows_for_key.iterrows():
                qc_rows.append({
                    "file_id": row["file_id"], "level": "visit", "path": row["path"],
                    "stage": "rpt",
                    "code": "visit_label_collision" if distinguishable else "ambiguous_duplicate_rpt",
                    "message": (
                        f"{count} valid RPT files share ({cond}, {cell}, V{visit:02d}); "
                        + ("distinct timestamps permit chronological matching" if distinguishable
                           else "timestamps do not distinguish the measurements")
                    ),
                    "severity": "warning" if distinguishable else "error",
                    "row": None, "column": None, "value": None,
                    "exception_type": None,
                    "action": ("preserve_raw_visit_and_match_by_time" if distinguishable
                               else "block_target_construction"),
                })
        qc = pd.DataFrame(qc_rows)

    if not manifest.empty:
        manifest["qc_status"] = "valid"
        if not qc.empty:
            warning_ids = set(qc.loc[qc["severity"] == "warning", "file_id"])
            error_ids = set(qc.loc[qc["severity"] == "error", "file_id"])
            manifest.loc[manifest["file_id"].isin(warning_ids), "qc_status"] = "warning"
            manifest.loc[manifest["file_id"].isin(error_ids), "qc_status"] = "error"
        manifest.loc[manifest["parse_status"] != "valid", "qc_status"] = "error"

    cell_metadata = (
        manifest[["condition", "cell", "DOD_pct", "C_ch", "C_dis", "T_degC"]]
        .drop_duplicates()
        .sort_values(["condition", "cell"])
        .reset_index(drop=True)
    ) if not manifest.empty else _empty(["condition", "cell"])
    if not cell_metadata.empty:
        cell_metadata["replicate"] = cell_metadata.groupby("condition").cumcount() + 1
        cell_metadata["terminal_outcome"] = pd.NA
        cell_metadata["terminal_outcome_status"] = "unverified"
    events = _empty([
        "condition", "cell", "event_type", "event_observed", "event_time",
        "event_EFC", "censor_time", "censor_EFC", "evidence_source", "qc_status",
    ])

    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(manifest, output / "file_manifest.parquet")
    _atomic_parquet(steps_df if not steps_df.empty else _empty(["file_id", "step_index"]),
                    output / "steps.parquet")
    _atomic_parquet(cycles if not cycles.empty else _empty(["file_id", "cycle_in_file", "cycle_complete"]),
                    output / "cycles.parquet")
    _atomic_parquet(rpt if not rpt.empty else _empty(["file_id", "reference_capacity_Ah", "rpt_qc_status"]),
                    output / "rpt_measurements.parquet")
    _atomic_parquet(qc if not qc.empty else _empty(["file_id", "level", "code", "severity", "action"]),
                    output / "qc_flags.parquet")
    _atomic_parquet(cell_metadata, output / "cell_metadata.parquet")
    _atomic_parquet(events, output / "events.parquet")
    _atomic_json({
        "dataset_root": str(root),
        "scope": "Standard cycling only",
        "file_count": int(len(manifest)),
        "parse_valid_count": int((manifest["parse_status"] == "valid").sum()) if not manifest.empty else 0,
        "complete_cycle_count": int(cycles["cycle_complete"].sum()) if not cycles.empty else 0,
        "valid_rpt_count": int((rpt["rpt_qc_status"] == "valid").sum()) if not rpt.empty else 0,
        "config_protocol": protocol,
        "scientific_blockers": [
            "Confirm the inferred 0.5C reference-discharge protocol rule against dataset documentation.",
            "Provide authoritative CID/gassing outcomes and censoring evidence before classification.",
            "Approve a sustained capacity-EOL definition before survival target construction.",
        ],
    }, output / "build_manifest.json")
    return CanonicalBuildResult(manifest, steps_df, cycles, rpt, qc)
