"""Model registry entry point."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from ..io.loaders import load_config
from .classifiers import NEEDS_SCALING as CLF_NEEDS_SCALING
from .classifiers import build_classifiers
from .regressors import NEEDS_SCALING as REG_NEEDS_SCALING
from .regressors import build_regressors, build_tuned_regressors


def regressors() -> Dict[str, Any]:
    return build_regressors()


def tuned_regressors() -> Dict[str, Any]:
    return build_tuned_regressors()


def classifiers() -> Dict[str, Any]:
    return build_classifiers()


def needs_scaling(name: str, kind: str) -> bool:
    if kind == "regression":
        return name in REG_NEEDS_SCALING
    return name in CLF_NEEDS_SCALING


def _validate_allowlist(names: Iterable[Any], available: Mapping[str, Any],
                        setting: str) -> List[str]:
    resolved: List[str] = []
    if names is None:
        raise ValueError(f"{setting} must be a non-empty list of model names")
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{setting} must be a list, not a scalar")
    for entry in names:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{setting} contains a non-string or empty entry: {entry!r}")
        resolved.append(entry)
    if not resolved:
        raise ValueError(f"{setting} must be a non-empty list of model names")
    duplicates = sorted({name for name in resolved if resolved.count(name) > 1})
    if duplicates:
        raise ValueError(f"{setting} contains duplicate names: {duplicates}")
    unknown = [name for name in resolved if name not in available]
    if unknown:
        raise ValueError(
            f"{setting} contains unknown names: {unknown}. "
            f"Available: {sorted(available)}"
        )
    return resolved


def _apply_n_jobs(estimators: Dict[str, Any], model_n_jobs: int | None) -> None:
    if model_n_jobs is None:
        return
    for estimator in estimators.values():
        params = getattr(estimator, "get_params", None)
        if not callable(params):
            continue
        if "n_jobs" in params(deep=False):
            try:
                estimator.set_params(n_jobs=int(model_n_jobs))
            except (TypeError, ValueError):
                continue


_OPTIONAL_REGRESSORS = {"XGBoost", "LightGBM", "CatBoost"}


def enabled_regressors(cfg: Mapping[str, Any] | None = None,
                       model_n_jobs: int | None = None) -> Dict[str, Any]:
    """Return the ordered dict of regressors permitted by ``models.enabled_regressors``.

    - ``cfg`` defaults to ``load_config()``; pass an override for tests.
    - ``model_n_jobs`` forces ``n_jobs`` on every estimator that exposes it, to
      prevent nested parallelism when the outer experiment loop is parallel.
    - Optional regressors (XGBoost, LightGBM, CatBoost) are skipped with a
      warning when the backing package is not installed, rather than aborting.
    """
    if cfg is None:
        cfg = load_config()
    model_cfg = (cfg.get("models") or {}) if isinstance(cfg, Mapping) else {}
    names = model_cfg.get("enabled_regressors")
    available = build_regressors()
    # Optional packages may be absent from the registry when not installed.
    # Warn and skip rather than aborting the pipeline.
    filtered_names: List[str] = []
    if names is not None and not isinstance(names, (str, bytes)):
        for name in names:
            if name in _OPTIONAL_REGRESSORS and name not in available:
                print(f"[registry] WARNING: {name} requested but package not installed; skipping.")
            else:
                filtered_names.append(name)
        names = filtered_names
    resolved = _validate_allowlist(names, available, "models.enabled_regressors")
    result: Dict[str, Any] = {name: available[name] for name in resolved}
    _apply_n_jobs(result, model_n_jobs)
    return result


def enabled_tuned_regressors(cfg: Mapping[str, Any] | None = None,
                             model_n_jobs: int | None = None) -> Dict[str, Any]:
    """Same as enabled_regressors() but every model is wrapped in GridSearchCV.

    Uses the same allowlist from ``models.enabled_regressors`` in config so
    you can run tuned and untuned experiments on the same model set.
    """
    if cfg is None:
        cfg = load_config()
    model_cfg = (cfg.get("models") or {}) if isinstance(cfg, Mapping) else {}
    names = model_cfg.get("enabled_regressors")
    available = build_tuned_regressors()
    filtered_names: List[str] = []
    if names is not None and not isinstance(names, (str, bytes)):
        for name in names:
            if name in _OPTIONAL_REGRESSORS and name not in available:
                print(f"[registry] WARNING: {name} requested but package not installed; skipping.")
            else:
                filtered_names.append(name)
        names = filtered_names
    resolved = _validate_allowlist(names, available, "models.enabled_regressors")
    result: Dict[str, Any] = {name: available[name] for name in resolved}
    _apply_n_jobs(result, model_n_jobs)
    return result


def enabled_classifiers(cfg: Mapping[str, Any] | None = None,
                        model_n_jobs: int | None = None) -> Dict[str, Any]:
    """Return the ordered dict of classifiers permitted by ``models.enabled_classifiers``.

    Falls back to the full classifier registry when the config key is absent,
    preserving prior behaviour for anyone who has not opted into the allowlist.
    """
    if cfg is None:
        cfg = load_config()
    model_cfg = (cfg.get("models") or {}) if isinstance(cfg, Mapping) else {}
    names = model_cfg.get("enabled_classifiers")
    available = build_classifiers()
    if names is None:
        _apply_n_jobs(available, model_n_jobs)
        return available
    resolved = _validate_allowlist(names, available, "models.enabled_classifiers")
    filtered: Dict[str, Any] = {name: available[name] for name in resolved}
    _apply_n_jobs(filtered, model_n_jobs)
    return filtered
