"""Shared PINN utilities: seeding, device selection, atomic IO, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seeds(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) reproducibly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(preferred: str | None = None) -> str:
    """Return 'cuda' if available and not forbidden, else 'cpu'."""
    try:
        import torch
    except Exception:
        return "cpu"
    if preferred in {"cpu", "cuda"}:
        if preferred == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_memory_mb() -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) for the current CUDA device.

    Returns (0.0, 0.0) when CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            float(torch.cuda.memory_allocated() / (1024 * 1024)),
            float(torch.cuda.memory_reserved() / (1024 * 1024)),
        )
    except Exception:
        return 0.0, 0.0


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_parquet(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_write_torch(state: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run_fingerprint(*, architecture: str, preprocessing: str, target: str | None,
                    fold: int, seed: int, extra: dict[str, Any] | None = None) -> str:
    """Deterministic identifier for one architecture x preprocessing x fold x seed run."""
    payload: dict[str, Any] = {
        "architecture": architecture,
        "preprocessing": preprocessing,
        "target": target or "capacity_state",
        "fold": int(fold),
        "seed": int(seed),
    }
    if extra:
        payload.update(extra)
    return hash_payload(payload)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def finite_or_raise(name: str, values) -> None:
    import torch

    if isinstance(values, torch.Tensor):
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"{name} contains nonfinite values")
    else:
        arr = np.asarray(values)
        if arr.size and not np.isfinite(arr).all():
            raise FloatingPointError(f"{name} contains nonfinite values")


def flatten_iter(items: Iterable[Iterable[Any]]) -> list[Any]:
    out: list[Any] = []
    for sub in items:
        out.extend(sub)
    return out
