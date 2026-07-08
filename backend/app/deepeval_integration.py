"""Centralized DeepEval helpers for optional runtime status checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

from .config import Settings


def deepeval_sdk_installed() -> bool:
    """Return whether the DeepEval package is importable in this runtime."""

    return find_spec("deepeval") is not None


def deepeval_package_version() -> str:
    """Return the installed DeepEval version when available."""

    try:
        return version("deepeval")
    except PackageNotFoundError:
        return ""


def deepeval_enabled(settings: Settings) -> bool:
    """Return whether DeepEval support is enabled for this project."""

    return bool(getattr(settings, "deepeval_enabled", True))


def deepeval_cloud_configured(settings: Settings) -> bool:
    """Return whether Confident AI upload credentials are configured."""

    return bool(str(getattr(settings, "confident_api_key", "") or "").strip())
