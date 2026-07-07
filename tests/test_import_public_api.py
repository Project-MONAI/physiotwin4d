"""Public import smoke tests."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def test_public_api_exports_are_importable() -> None:
    """Every name in physiotwin4d.__all__ resolves from the package."""
    package = importlib.import_module("physiotwin4d")

    public_names = getattr(package, "__all__", None)
    assert isinstance(public_names, Sequence), (
        "physiotwin4d.__all__ should be a sequence of public export names"
    )
    assert public_names, "physiotwin4d.__all__ should not be empty"

    for name in public_names:
        assert hasattr(package, name), f"Missing public export: {name}"
