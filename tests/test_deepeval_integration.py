from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

from backend.app import deepeval_integration as di


def test_deepeval_sdk_installed_reflects_importability(monkeypatch):
    monkeypatch.setattr(di, "find_spec", lambda _name: object())
    assert di.deepeval_sdk_installed() is True

    monkeypatch.setattr(di, "find_spec", lambda _name: None)
    assert di.deepeval_sdk_installed() is False


def test_deepeval_package_version_handles_missing_package(monkeypatch):
    monkeypatch.setattr(di, "version", lambda _name: "4.0.2")
    assert di.deepeval_package_version() == "4.0.2"

    def raise_missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(di, "version", raise_missing)
    assert di.deepeval_package_version() == ""


def test_deepeval_project_flags_derive_from_settings():
    assert di.deepeval_enabled(SimpleNamespace(deepeval_enabled=True)) is True
    assert di.deepeval_enabled(SimpleNamespace(deepeval_enabled=False)) is False
    assert di.deepeval_cloud_configured(SimpleNamespace(confident_api_key="confident_us_demo")) is True
    assert di.deepeval_cloud_configured(SimpleNamespace(confident_api_key="  ")) is False
