from __future__ import annotations

from tests.helpers import requires_torch


def test_attention_backward_runs() -> None:
    requires_torch()

    import torch

    from novadit.modules import NovaDiTAttention

    x = torch.randn(2, 8, 32, requires_grad=True)
    module = NovaDiTAttention(hidden_size=32, num_heads=4)
    loss = module(x).float().pow(2).mean()
    loss.backward()
    assert x.grad is not None
