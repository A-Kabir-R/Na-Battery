"""Skip the two known-bad .ird files from any downstream processing."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..io.loaders import load_config


def bad_filenames() -> List[str]:
    return load_config()["cleaning"]["skip_files"]


def skip_bad_paths(paths: Iterable[str | Path]) -> List[Path]:
    bad = set(bad_filenames())
    return [Path(p) for p in paths if Path(p).name not in bad]


def filter_inventory(inv: pd.DataFrame, path_col: str = "path") -> pd.DataFrame:
    bad = set(bad_filenames())
    return inv[~inv[path_col].apply(lambda p: Path(p).name in bad)].reset_index(drop=True)
