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
    except Exception:
        pass
    # Backend flags are deliberately NOT set here. They are global state, and
    # setting them from two places (this function and a module-level block in
    # trainer.py) made the effective configuration depend on import-versus-call
    # order. Use configure_torch_backend() explicitly instead.


def configure_torch_backend(*, deterministic: bool) -> dict[str, object]:
    """Set every backend flag that affects reproducibility, in one place.

    ``deterministic=True`` is the publication mode: reproducible bitwise on
    fixed hardware, at some throughput cost. ``False`` is the fast mode, which
    enables TF32 and cuDNN autotuning -- both change low-order bits, so runs are
    not comparable across machines or even across repeated runs.

    Returns the resulting state so callers can record it in a run fingerprint;
    a reproducibility claim that is not recorded is not verifiable.
    """
    state: dict[str, object] = {"deterministic_mode": bool(deterministic)}
    try:
        import torch
    except Exception:
        return state
    allow_tf32 = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("highest" if deterministic else "high")
    except AttributeError:
        pass
    state.update({
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    })
    return state


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
