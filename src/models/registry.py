"""Model registry entry point."""
from __future__ import annotations

from .classifiers import NEEDS_SCALING as CLF_NEEDS_SCALING
from .classifiers import build_classifiers
from .regressors import NEEDS_SCALING as REG_NEEDS_SCALING
from .regressors import build_regressors


def regressors():
    return build_regressors()


def classifiers():
    return build_classifiers()


def needs_scaling(name: str, kind: str) -> bool:
    if kind == "regression":
        return name in REG_NEEDS_SCALING
    return name in CLF_NEEDS_SCALING
