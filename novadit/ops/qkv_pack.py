from __future__ import annotations

import torch
import torch.nn.functional as F

from . import novadit_cuda


def qkv_pack_ref(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, three_hidden = qkv.shape
    hidden = num_heads * head_dim
    if three_hidden != 3 * hidden:
        raise ValueError(f"Expected last dim {3 * hidden}, got {three_hidden}")

    qkv_view = qkv.view(bsz, seq_len, 3, num_heads, head_dim)
    q = qkv_view[:, :, 0].contiguous()
    k = qkv_view[:, :, 1].contiguous()
    v = qkv_view[:, :, 2].contiguous()
    return q, k, v


def fused_qkv_pack(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if novadit_cuda is not None and qkv.is_cuda:
        return novadit_cuda.fused_qkv_pack_forward(qkv, num_heads, head_dim)
    return qkv_pack_ref(qkv, num_heads, head_dim)


def fused_qkv_linear_pack(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = F.linear(x, weight, bias)
    return fused_qkv_pack(qkv, num_heads=num_heads, head_dim=head_dim)
