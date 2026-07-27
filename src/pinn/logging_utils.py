"""Structured logging for the NaPINN-Q pipeline.

Every script and the trainer route through :func:`get_logger`, which attaches
one console handler and one rotating-per-run file handler. Log files land in
``artifacts/results/pinn_preprocessing_study/logs/`` by default and record
- configuration snapshot,
- dataset shape / feature counts / fold sizes,
- model parameter counts,
- per-epoch summaries + validation metrics,
- physics metrics,
- failure tracebacks.

The file handler writes JSONL alongside the plain-text log so downstream
debug scripts can parse structured events without regex.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAMESPACE = "napinn_q"
_INITIALIZED: dict[str, logging.Logger] = {}


class JsonLineFormatter(logging.Formatter):
    """Formatter that emits one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(script_name: str, log_dir: Path | str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to console + ``<log_dir>/<script_name>_<ts>.log``.

    Repeated calls with the same ``script_name`` reuse the same logger without
    duplicating handlers.
    """
    key = f"{LOGGER_NAMESPACE}.{script_name}"
    if key in _INITIALIZED:
        return _INITIALIZED[key]

    logger = logging.getLogger(key)
    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        text_path = log_dir / f"{script_name}_{timestamp}.log"
        jsonl_path = log_dir / f"{script_name}_{timestamp}.jsonl"
        text_handler = RotatingFileHandler(
            text_path, maxBytes=25 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        text_handler.setFormatter(logging.Formatter(
            "%(asctime)sZ [%(levelname)s] %(name)s :: %(message)s",
        ))
        text_handler.setLevel(level)
        logger.addHandler(text_handler)

        jsonl_handler = RotatingFileHandler(
            jsonl_path, maxBytes=25 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        jsonl_handler.setFormatter(JsonLineFormatter())
        jsonl_handler.setLevel(level)
        logger.addHandler(jsonl_handler)

        logger.info("logging_initialized",
                    extra={"structured": {"text_log": str(text_path),
                                           "jsonl_log": str(jsonl_path)}})

    _INITIALIZED[key] = logger
    return logger


def log_event(logger: logging.Logger, level: int, message: str,
              **fields: Any) -> None:
    """Emit a structured event; fields land in the JSONL log verbatim."""
    logger.log(level, message, extra={"structured": fields})


def log_config_snapshot(logger: logging.Logger, cfg: dict[str, Any]) -> None:
    """Log the full PINN configuration in one structured event."""
    pinn = cfg.get("pinn", {})
    log_event(
        logger, logging.INFO, "pinn_config_snapshot",
        model_name=pinn.get("model_name"),
        architectures=pinn.get("architectures"),
        preprocessing=pinn.get("preprocessing"),
        seeds=pinn.get("seeds"),
        stress=pinn.get("stress"),
        model=pinn.get("model"),
        collocation=pinn.get("collocation"),
        quadrature=pinn.get("quadrature"),
        training=pinn.get("training"),
        losses=pinn.get("losses"),
        bounds=pinn.get("bounds"),
        monotonicity=pinn.get("monotonicity"),
        curriculum=pinn.get("curriculum"),
    )


def log_dataset_summary(logger: logging.Logger, dataset, frame,
                         preprocessing: str) -> None:
    """Log dataset shape, feature list, split-role counts, stress range."""
    log_event(
        logger, logging.INFO, "dataset_summary",
        preprocessing=preprocessing,
        n_anchor_rows=int(len(frame)),
        n_cells=int(frame["cell"].nunique()) if "cell" in frame.columns else 0,
        n_conditions=int(frame["condition"].nunique()) if "condition" in frame.columns else 0,
        n_features=len(dataset.feature_columns),
        feature_columns=list(dataset.feature_columns),
        stress_column=dataset.stress_column,
        horizon_column=dataset.horizon_column,
        stress_min=float(frame["stress_current"].min()),
        stress_max=float(frame["stress_current"].max()),
        stress_mean=float(frame["stress_current"].mean()),
        horizon_min=float(frame["stress_delta"].min()),
        horizon_max=float(frame["stress_delta"].max()),
        u_current_min=float(frame["u_current"].min()),
        u_current_max=float(frame["u_current"].max()),
        u_true_min=float(frame["u_true_next"].min()),
        u_true_max=float(frame["u_true_next"].max()),
    )


def log_fold_summary(logger: logging.Logger, *, architecture: str,
                     preprocessing: str, fold: int, seed: int,
                     outer_train_cells: int, outer_val_cells: int,
                     inner_train_cells: int, inner_val_cells: int,
                     n_rows_train: int, n_rows_val: int,
                     trainable_parameters: int, u_max: float,
                     epsilon_rec: float, device: str) -> None:
    log_event(
        logger, logging.INFO, "fold_summary",
        architecture=architecture, preprocessing=preprocessing,
        fold=fold, seed=seed,
        outer_train_cells=outer_train_cells,
        outer_val_cells=outer_val_cells,
        inner_train_cells=inner_train_cells,
        inner_val_cells=inner_val_cells,
        n_rows_train=n_rows_train,
        n_rows_val=n_rows_val,
        trainable_parameters=trainable_parameters,
        u_max=u_max,
        epsilon_rec=epsilon_rec,
        device=device,
    )


def log_epoch_summary(logger: logging.Logger, row: dict[str, Any]) -> None:
    """Emit one structured epoch record (skip pretty printing on console)."""
    log_event(logger, logging.DEBUG, "epoch", **row)


def log_failure(logger: logging.Logger, message: str, exc: BaseException,
                **fields: Any) -> None:
    logger.exception(message, extra={"structured": {**fields,
                                                     "error_type": type(exc).__name__,
                                                     "error_message": str(exc)}})
