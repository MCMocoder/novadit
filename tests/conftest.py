from __future__ import annotations

import importlib.util

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


def requires_torch():
    if not HAS_TORCH:
        pytest.skip("torch is not installed in the current environment")
