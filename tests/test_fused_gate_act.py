from __future__ import annotations

from tests.helpers import requires_torch


def test_gate_act_shape() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_gate_act

    x = torch.randn(2, 8, 64)
    y = fused_gate_act(x, activation="silu")
    assert y.shape == (2, 8, 32)
