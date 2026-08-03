"""Backend reproducibility flags must be explicit, not import-order dependent.

`trainer.py` used to set TF32 and `cudnn.benchmark=True` at module import while
`set_seeds()` set the opposite when it ran, so whichever executed last silently
decided whether a run was reproducible.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.utils import configure_torch_backend, set_seeds


def test_set_seeds_does_not_touch_backend_flags():
    configure_torch_backend(deterministic=False)
    before = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
    )
    set_seeds(123)
    after = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
    )
    assert before == after, "set_seeds must seed only; backend state is configured separately"


def test_importing_trainer_does_not_mutate_backend_flags():
    """Checked in a subprocess so the import is genuinely first-time.

    An in-process ``importlib.reload`` would not prove anything -- the module is
    already imported by the time the suite runs -- and it rebinds every class in
    the module, which breaks identity for tests that ran before it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import torch
        before = (torch.backends.cudnn.benchmark,
                  torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32)
        import src.pinn.trainer  # noqa: F401
        after = (torch.backends.cudnn.benchmark,
                 torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        assert before == after, f"import mutated backend state: {before} -> {after}"
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@pytest.mark.parametrize("deterministic", [True, False])
def test_configure_torch_backend_is_internally_consistent(deterministic):
    state = configure_torch_backend(deterministic=deterministic)
    assert state["deterministic_mode"] is deterministic
    # Deterministic mode must disable every source of run-to-run variation;
    # fast mode must enable them. A mixture is the bug this replaces.
    assert state["cudnn_deterministic"] is deterministic
    assert state["cudnn_benchmark"] is (not deterministic)
    assert state["matmul_allow_tf32"] is (not deterministic)
    assert state["cudnn_allow_tf32"] is (not deterministic)


def test_trainer_config_defaults_to_deterministic():
    from src.pinn.trainer import TrainerConfig

    config = TrainerConfig(architecture="a", preprocessing="unified", fold=0, seed=42)
    assert config.deterministic is True, "publication runs must default to reproducible"
