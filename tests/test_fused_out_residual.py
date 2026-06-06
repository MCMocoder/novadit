from __future__ import annotations

from tests.helpers import requires_torch


def test_out_residual_shape() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_out_residual

    x = torch.randn(2, 8, 16)
    residual = torch.randn(2, 8, 16)
    bias = torch.randn(16)
    y = fused_out_residual(x, residual, bias)
    assert y.shape == x.shape
