from __future__ import annotations

from tests.helpers import requires_torch


def test_qkv_pack_shapes() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_qkv_pack

    qkv = torch.randn(2, 8, 3 * 32)
    q, k, v = fused_qkv_pack(qkv, num_heads=4, head_dim=8)
    assert q.shape == (2, 8, 4, 8)
    assert k.shape == (2, 8, 4, 8)
    assert v.shape == (2, 8, 4, 8)
