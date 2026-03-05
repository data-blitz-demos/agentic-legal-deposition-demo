# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import get_settings


def pytest_configure() -> None:
    """Ensure report output directories exist before pytest-html/junit write artifacts."""

    Path("reports").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
