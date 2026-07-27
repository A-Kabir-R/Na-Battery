"""Checkpoint round-trip test for the trainer state dict."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ
from src.pinn.utils import atomic_write_torch


def test_atomic_torch_roundtrip(tmp_path: Path) -> None:
    model = NaPINNQ(feature_dim=3, solution_hidden_dims=(8,), rate_hidden_dims=(4,))
    ckpt = tmp_path / "best_model.pt"
    atomic_write_torch({"model": model.state_dict(), "epoch": 42}, ckpt)
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 42
    new_model = NaPINNQ(feature_dim=3, solution_hidden_dims=(8,), rate_hidden_dims=(4,))
    new_model.load_state_dict(payload["model"])
    for (a, va), (b, vb) in zip(model.state_dict().items(),
                                 new_model.state_dict().items()):
        assert a == b
        assert torch.allclose(va, vb)
