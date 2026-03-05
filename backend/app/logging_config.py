# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""Central application logging configuration.

This module configures one process-wide stream handler and exposes a stable
logger namespace for backend modules. The implementation is intentionally
small and idempotent so tests and repeated imports do not accumulate
duplicate handlers.
"""

import logging
from pathlib import Path
import sys

from .metrics import record_log_event


APP_LOGGER_NAME = "legal_deposition_demo"


class _PrometheusLogHandler(logging.Handler):
    """Lightweight handler that mirrors log-level counts into Prometheus."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record_log_event(record.levelname)
        except Exception:
            return


def _resolve_log_level(value: str | int | None) -> int:
    """Normalize a string or numeric log level into a stdlib logging value."""

    if isinstance(value, int):
        return value
    candidate = str(value or "").strip().upper()
    if not candidate:
        return logging.INFO
    return int(getattr(logging, candidate, logging.INFO))


def configure_application_logging(
    level: str | int | None = "INFO",
    log_file_path: str | None = None,
) -> logging.Logger:
    """Install one stdout handler on the root logger and return the app logger."""

    root_logger = logging.getLogger()
    resolved_level = _resolve_log_level(level)

    handler = next(
        (
            item
            for item in root_logger.handlers
            if getattr(item, "_legal_deposition_demo_handler", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler._legal_deposition_demo_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        root_logger.addHandler(handler)

    handler.setLevel(resolved_level)
    root_logger.setLevel(resolved_level)

    if log_file_path:
        file_handler = next(
            (
                item
                for item in root_logger.handlers
                if getattr(item, "_legal_deposition_demo_file_handler", False)
            ),
            None,
        )
        if file_handler is None:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler._legal_deposition_demo_file_handler = True  # type: ignore[attr-defined]
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S%z",
                )
            )
            root_logger.addHandler(file_handler)
        file_handler.setLevel(resolved_level)

    prometheus_handler = next(
        (
            item
            for item in root_logger.handlers
            if getattr(item, "_legal_deposition_demo_prometheus_handler", False)
        ),
        None,
    )
    if prometheus_handler is None:
        prometheus_handler = _PrometheusLogHandler()
        prometheus_handler._legal_deposition_demo_prometheus_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(prometheus_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(resolved_level)

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(resolved_level)
    return app_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an application-scoped logger for one backend module."""

    normalized_name = str(name or "").strip()
    if not normalized_name:
        return logging.getLogger(APP_LOGGER_NAME)
    if normalized_name.startswith(APP_LOGGER_NAME):
        return logging.getLogger(normalized_name)
    return logging.getLogger(f"{APP_LOGGER_NAME}.{normalized_name}")
