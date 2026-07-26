from pathlib import Path

import pytest

from src.io.canonical import discover_standard_cycling_files


def test_discovery_accepts_extracted_root_name_when_dod_folders_are_direct(tmp_path: Path):
    root = tmp_path / "dataset"
    cycling = root / "DOD20_2C2C_25degC" / "Cycling_Periods"
    cycling.mkdir(parents=True)
    raw = cycling / "S2000FN6H3_CYC_at25degC_V01_4924.ird"
    raw.touch()
    assert discover_standard_cycling_files(root) == [raw.resolve()]


def test_discovery_rejects_root_without_standard_cycling_conditions(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(ValueError, match="directly contain"):
        discover_standard_cycling_files(root)
