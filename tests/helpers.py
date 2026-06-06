from __future__ import annotations

import importlib.util

import pytest


def requires_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in the current environment")
