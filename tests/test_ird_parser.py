from pathlib import Path

import numpy as np
import pytest

from src.io.ird_parser import IRDParseError, _parse_time, parse_ird


def _write_ird(path: Path, rows: list[str], *, marker: bool = True) -> None:
    header = [
        "Cellname: ' test-cell'",
        "Starttime: '2024-01-02 03:04:05'",
        "Testname: ' test-run'",
        "%%%",
        "Time,Voltage,Current,Ah_counter,Temperature,StepID",
        "timedelta,V,A,Ah,C,",
    ]
    if marker:
        header.extend(["###", "Time,Voltage,Current,Ah_counter,Temperature,StepID"])
    path.write_text("\n".join([*header, *rows]) + "\n", encoding="latin-1")


def test_parse_time_supports_mixed_numeric_and_timedelta_values():
    import pandas as pd

    values = pd.Series(["1.5", "0 days 00:00:02.500", "1 day  00:00:03.0", None])
    parsed = _parse_time(values)
    assert parsed.iloc[:3].tolist() == pytest.approx([1.5, 2.5, 86403.0])
    assert np.isnan(parsed.iloc[3])


def test_parse_ird_returns_metadata_and_timestamp_qc(tmp_path: Path):
    path = tmp_path / "sample.ird"
    _write_ird(path, [
        "0,3.0,0.0,0.0,25.0,1",
        "0 days 00:00:01,3.1,0.6,0.0001,25.1,2",
        "0 days 00:00:01,3.2,0.6,0.0002,25.2,2",
    ])
    result = parse_ird(path, strict=False)
    assert result.data is not None
    assert result.metadata["starttime"].isoformat() == "2024-01-02T03:04:05"
    assert result.qc["duplicate_timestamp_count"] == 1
    assert any(issue.code == "duplicate_time" for issue in result.issues)


def test_missing_data_marker_is_a_structured_error(tmp_path: Path):
    path = tmp_path / "missing-marker.ird"
    _write_ird(path, ["0,3.0,0.0,0.0,25.0,1"], marker=False)
    with pytest.raises(IRDParseError) as error:
        parse_ird(path)
    assert error.value.issue.code == "missing_data_marker"


def test_empty_file_is_a_structured_error(tmp_path: Path):
    path = tmp_path / "empty.ird"
    path.touch()
    result = parse_ird(path, strict=False)
    assert result.data is None
    assert result.qc["parse_error_code"] == "empty_file"
