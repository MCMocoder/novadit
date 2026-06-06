from __future__ import annotations

import torch

from . import novadit_cuda


def out_residual_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None:
        return x + residual + bias.view(1, 1, -1)
    return x + residual


def fused_out_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if novadit_cuda is not None and x.is_cuda:
        return novadit_cuda.fused_out_residual_forward(x, residual, bias)
    return out_residual_ref(x, residual, bias)
