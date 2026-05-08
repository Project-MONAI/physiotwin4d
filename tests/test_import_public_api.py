"""Public import smoke tests."""

from __future__ import annotations

import importlib


def test_public_api_exports_are_importable() -> None:
    """Every name in physiomotion4d.__all__ resolves from the package."""
    package = importlib.import_module("physiomotion4d")

    public_names = getattr(package, "__all__")
    assert public_names, "physiomotion4d.__all__ should not be empty"

    for name in public_names:
        assert hasattr(package, name), f"Missing public export: {name}"
