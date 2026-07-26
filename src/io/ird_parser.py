"""Structured parser for the battery cycler ``.ird`` export format."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<cell>S\d{4}[A-Z0-9]+)_(?P<test>[A-Z]+)_at(?P<rpt_T>-?\d+)degC_"
    r"V(?P<visit>\d+)_(?P<testid>\d+)\.ird$"
)

REQUIRED_COLS = ["Time", "Voltage", "Current", "Ah_counter", "Temperature", "StepID"]
COLUMN_ALIASES = {
    "time": "Time",
    "voltage": "Voltage",
    "current": "Current",
    "ah_counter": "Ah_counter",
    "ahcounter": "Ah_counter",
    "temperature": "Temperature",
    "stepid": "StepID",
    "step_id": "StepID",
}


@dataclass(frozen=True)
class IRDParseIssue:
    """Serializable parser or signal-quality issue."""

    path: str
    stage: str
    code: str
    message: str
    severity: str = "error"
    row: int | None = None
    column: str | None = None
    value: str | None = None
    exception_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IRDParseError(ValueError):
    """Raised when an IRD file cannot satisfy the parser contract."""

    def __init__(self, issue: IRDParseIssue):
        self.issue = issue
        super().__init__(issue.message)


@dataclass
class IRDParseResult:
    data: pd.DataFrame | None
    metadata: dict[str, Any]
    issues: list[IRDParseIssue]
    qc: dict[str, Any]


def parse_condition(cond: str) -> Dict[str, Any]:
    m = re.match(
        r"DOD(?P<dod>\d+)_(?P<cch>[\d.]+)C(?P<cdis>[\d.]+)C_(?P<T>-?\d+)degC$",
        cond,
    )
    if not m:
        return {"DOD_pct": np.nan, "C_ch": np.nan, "C_dis": np.nan, "T_degC": np.nan}
    return {
        "DOD_pct": int(m["dod"]),
        "C_ch": float(m["cch"]),
        "C_dis": float(m["cdis"]),
        "T_degC": int(m["T"]),
    }


def _issue(path: Path, stage: str, code: str, message: str, **kwargs: Any) -> IRDParseIssue:
    return IRDParseIssue(path=str(path), stage=stage, code=code, message=message, **kwargs)


def _header(path: Path) -> tuple[int, dict[str, str]]:
    """Return the data-header offset and normalized file metadata."""
    metadata: dict[str, str] = {}
    try:
        with path.open("r", errors="replace", encoding="latin-1") as handle:
            for i, line in enumerate(handle):
                stripped = line.strip()
                if stripped.startswith("###"):
                    return i + 1, metadata
                if ":" in line and not stripped.startswith(("%%%", "###")):
                    key, value = line.split(":", 1)
                    if key.strip() in {"Cellname", "Starttime", "Testname"}:
                        metadata[key.strip().lower()] = value.strip().strip("'").strip()
    except OSError as exc:
        raise IRDParseError(
            _issue(path, "open", "file_open_error", f"could not open {path}: {exc}",
                   exception_type=type(exc).__name__)
        ) from exc
    raise IRDParseError(
        _issue(path, "header", "missing_data_marker", f"missing ### data marker in {path}")
    )


def find_data_offset(path: Path) -> int:
    """Compatibility helper that now fails instead of guessing offset zero."""
    offset, _ = _header(Path(path))
    return offset


_TD_RE = re.compile(
    r"^\s*(?:(?P<days>\d+)\s*days?\s*)?"
    r"(?P<hours>\d+)\s*:\s*(?P<minutes>\d+)\s*:\s*(?P<seconds>\d+(?:\.\d*)?)\s*$",
    re.IGNORECASE,
)


def _td_str_to_sec(value: Any) -> float:
    """Parse a numeric or timedelta-like scalar to seconds."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = _TD_RE.match(text)
    if not match:
        return float("nan")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        return float("nan")
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def _parse_time(t: pd.Series) -> pd.Series:
    """Parse mixed numeric, timedelta, and datetime values element by element."""
    result = pd.Series(np.nan, index=t.index, dtype="float64")
    numeric = pd.to_numeric(t, errors="coerce")
    result.loc[numeric.notna()] = numeric.loc[numeric.notna()].astype(float)

    unresolved = result.isna() & t.notna()
    if unresolved.any():
        elapsed = t.loc[unresolved].map(_td_str_to_sec)
        result.loc[elapsed.index] = elapsed

    unresolved = result.isna() & t.notna()
    if unresolved.any():
        datetimes = pd.to_datetime(t.loc[unresolved], errors="coerce")
        if datetimes.notna().any():
            origin = datetimes.loc[datetimes.notna()].iloc[0]
            parsed = (datetimes - origin).dt.total_seconds()
            result.loc[parsed.index] = parsed
    return result


def _canonicalize_columns(columns: list[Any]) -> tuple[list[str], dict[str, str]]:
    mapped: list[str] = []
    aliases: dict[str, str] = {}
    for raw in columns:
        name = str(raw).strip()
        canonical = COLUMN_ALIASES.get(name.lower(), name)
        mapped.append(canonical)
        aliases[name] = canonical
    return mapped, aliases


def parse_ird(path: Path | str, *, strict: bool = True,
              max_invalid_fraction: float = 0.0,
              max_sampling_gap_s: float = 300.0) -> IRDParseResult:
    """Parse one file and return data, metadata, and explicit QC diagnostics.

    ``strict`` controls whether contract errors raise immediately. Signal QC
    findings remain in ``issues`` so canonical builders can retain the file.
    """
    path = Path(path)
    issues: list[IRDParseIssue] = []
    qc: dict[str, Any] = {"path": str(path), "parse_ok": False}
    try:
        if not path.exists():
            raise IRDParseError(_issue(path, "open", "file_not_found", f"file does not exist: {path}"))
        if path.stat().st_size == 0:
            raise IRDParseError(_issue(path, "open", "empty_file", f"empty IRD file: {path}"))
        offset, metadata = _header(path)
        try:
            raw = pd.read_csv(
                path,
                skiprows=offset,
                header=0,
                engine="c",
                low_memory=False,
                encoding="latin-1",
            )
        except Exception as exc:
            raise IRDParseError(
                _issue(path, "csv", "csv_parse_error", f"could not parse CSV data in {path}: {exc}",
                       exception_type=type(exc).__name__)
            ) from exc
        if raw.empty:
            raise IRDParseError(_issue(path, "csv", "empty_data", f"no sample rows in {path}"))

        original_columns = [str(c).strip() for c in raw.columns]
        raw.columns, aliases = _canonicalize_columns(list(raw.columns))
        missing = [column for column in REQUIRED_COLS if column not in raw.columns]
        if missing:
            raise IRDParseError(
                _issue(path, "schema", "missing_columns",
                       f"missing columns {missing} in {path.name}; have {original_columns}")
            )

        data = raw[REQUIRED_COLS].copy()
        data["Time"] = _parse_time(data["Time"])
        for column in ["Voltage", "Current", "Ah_counter", "Temperature"]:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
        step_numeric = pd.to_numeric(data["StepID"], errors="coerce")
        data["StepID"] = step_numeric.fillna(-1).astype("int32")

        invalid_counts: dict[str, int] = {}
        for column in REQUIRED_COLS:
            invalid = int(data[column].isna().sum())
            if column == "StepID":
                invalid = int((data[column] < 0).sum())
            invalid_counts[column] = invalid
            fraction = invalid / len(data)
            if invalid:
                severity = "error" if fraction > max_invalid_fraction else "warning"
                issues.append(
                    _issue(path, "values", "invalid_values",
                           f"{invalid}/{len(data)} invalid values in {column}", severity=severity,
                           column=column)
                )

        times = data["Time"].to_numpy(dtype=float)
        finite_times = times[np.isfinite(times)]
        if finite_times.size:
            dt = np.diff(finite_times)
            duplicate_count = int(np.sum(dt == 0))
            reset_count = int(np.sum(dt < 0))
            max_gap = float(np.max(dt)) if dt.size else 0.0
        else:
            duplicate_count = reset_count = 0
            max_gap = float("nan")
        if reset_count:
            issues.append(_issue(path, "time_qc", "time_reset",
                                 f"detected {reset_count} decreasing timestamp transitions"))
        if duplicate_count:
            issues.append(_issue(path, "time_qc", "duplicate_time",
                                 f"detected {duplicate_count} duplicate timestamp transitions",
                                 severity="warning"))
        if np.isfinite(max_gap) and max_gap > max_sampling_gap_s:
            issues.append(_issue(path, "time_qc", "sampling_gap",
                                 f"maximum sampling gap is {max_gap:.3f} s",
                                 severity="warning"))

        metadata = {
            **metadata,
            "starttime": pd.to_datetime(metadata.get("starttime"), errors="coerce"),
            "raw_columns": original_columns,
            "column_aliases": aliases,
        }
        qc.update({
            "parse_ok": not any(issue.severity == "error" for issue in issues),
            "row_count": int(len(data)),
            "invalid_counts": invalid_counts,
            "duplicate_timestamp_count": duplicate_count,
            "time_reset_count": reset_count,
            "max_sampling_gap_s": max_gap,
        })
        if strict and not qc["parse_ok"]:
            first_error = next(issue for issue in issues if issue.severity == "error")
            raise IRDParseError(first_error)
        return IRDParseResult(data=data, metadata=metadata, issues=issues, qc=qc)
    except IRDParseError as exc:
        issues.append(exc.issue)
        qc["parse_error_code"] = exc.issue.code
        if strict:
            raise
        return IRDParseResult(data=None, metadata={}, issues=issues, qc=qc)


def read_ird(path: Path | str) -> pd.DataFrame:
    """Compatibility API returning only the six canonical sample columns."""
    result = parse_ird(path, strict=True)
    assert result.data is not None
    return result.data[REQUIRED_COLS]
