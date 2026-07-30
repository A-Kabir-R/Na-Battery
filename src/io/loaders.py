"""Load the pre-computed statistics tables and provide config access.

Config supports two substitution forms:
  ${paths.a.b}         -> another node in the yaml
  ${env:VAR:-default}  -> os.environ["VAR"] if set, otherwise "default".
                         The default may itself contain ${...} references.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

_CONFIG_CACHE: Dict[str, Any] | None = None

_ENV_PAT = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\$\{[^{}]*\})*))?\}")
_REF_PAT = re.compile(r"\$\{([^${}][^}]*)\}")


def _sub_env(s: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(name, default)
    prev = None
    while prev != s:
        prev = s
        s = _ENV_PAT.sub(repl, s)
    return s


def _resolve_vars(node: Any, root: Dict[str, Any]) -> Any:
    if isinstance(node, str):
        # env first (may reference paths.* in its default), then paths.*
        def ref_repl(m: re.Match) -> str:
            keys = m.group(1).split(".")
            cur: Any = root
            for k in keys:
                cur = cur[k]
            return str(cur)

        prev = None
        while prev != node:
            prev = node
            node = _sub_env(node)
            node = _REF_PAT.sub(ref_repl, node)
        return node
    if isinstance(node, dict):
        return {k: _resolve_vars(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_vars(v, root) for v in node]
    return node


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_vars(cfg, cfg)
    _CONFIG_CACHE = cfg
    return cfg


def load_cycle_metrics() -> pd.DataFrame:
    cfg = load_config()
    canonical = Path(cfg["paths"]["canonical"]) / "cycles.parquet"
    if not canonical.exists():
        raise FileNotFoundError(
            f"canonical cycles are missing at {canonical}; run "
            "`python3 scripts/00_build_canonical.py` first"
        )
    return pd.read_parquet(canonical)


_DOD_ANCHOR = re.compile(r"(DOD\d+_[^/]+/(?:CU_at_[^/]+|Cycling_Periods)/[^/]+\.ird)$")


def _rewrite_paths(paths: pd.Series, dataset_root: str) -> pd.Series:
    """Rewrite each inventory path so it lives under `dataset_root`.

    inventory.csv was written on the laptop with absolute paths baked in.
    On the pod (or any machine with a different dataset location) we replace
    the prefix by extracting the DOD.../<file>.ird suffix and joining onto
    `dataset_root`. Falls through unchanged when the file already exists.
    """
    root = Path(dataset_root)

    def _one(p: str) -> str:
        if not isinstance(p, str):
            return p
        if Path(p).exists():
            return p
        m = _DOD_ANCHOR.search(p.replace("\\", "/"))
        if not m:
            return p
        return str(root / m.group(1))

    return paths.map(_one)


def load_inventory() -> pd.DataFrame:
    cfg = load_config()
    canonical = Path(cfg["paths"]["canonical"]) / "file_manifest.parquet"
    if canonical.exists():
        return pd.read_parquet(canonical)
    inv = pd.read_csv(cfg["paths"]["tables"]["inventory"])
    inv["path"] = _rewrite_paths(inv["path"], cfg["paths"]["dataset_root"])
    return inv


def load_rpt_measurements() -> pd.DataFrame:
    cfg = load_config()
    path = Path(cfg["paths"]["canonical"]) / "rpt_measurements.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"canonical RPT measurements are missing at {path}; run "
            "`python3 scripts/00_build_canonical.py` first"
        )
    return pd.read_parquet(path)


def load_steps_table() -> pd.DataFrame:
    """Canonical per-step table. Used by the unified DCIR extractor."""
    cfg = load_config()
    path = Path(cfg["paths"]["canonical"]) / "steps.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"canonical steps are missing at {path}; run "
            "`python3 scripts/00_build_canonical.py` first"
        )
    return pd.read_parquet(path)


def load_degradation_rates() -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(cfg["paths"]["tables"]["degradation_rates"])


def load_file_statistics() -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(cfg["paths"]["tables"]["file_statistics"])


def load_condition_summary() -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(cfg["paths"]["tables"]["condition_summary"])
