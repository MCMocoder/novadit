from __future__ import annotations

from tests.helpers import requires_torch


def test_precision_policy_returns_known_dtype() -> None:
    requires_torch()

    import torch

    from novadit.precision import autocast_dtype_for_training

    assert autocast_dtype_for_training() in {torch.float16, torch.bfloat16, torch.float32}
