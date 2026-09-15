from __future__ import annotations

import torch


def _broadcast_modulation(x: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    while tensor.dim() < x.dim():
        tensor = tensor.unsqueeze(1)
    return tensor


@torch.compile(dynamic=True)
def _adaln_compiled_fn(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    xhat = (x.float() - mean) * torch.rsqrt(var + eps)
    scale_b = _broadcast_modulation(x, scale).float()
    shift_b = _broadcast_modulation(x, shift).float()
    y = xhat * (1.0 + scale_b) + shift_b
    return y.to(dtype=x.dtype)


@torch.compile(dynamic=True)
def _adarmsnorm_compiled_fn(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = x.float() / rms
    scale_b = _broadcast_modulation(x, scale).float()
    shift_b = _broadcast_modulation(x, shift).float()
    y = xhat * (1.0 + scale_b) + shift_b
    return y.to(dtype=x.dtype)


def adaln_compile(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """torch.compile baseline for Ada LayerNorm.

    Uses torch.compile to generate Triton auto-fusion kernels.
    Supports autograd; both forward and backward are compiled.
    """
    return _adaln_compiled_fn(x, scale, shift, eps)


def adarmsnorm_compile(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """torch.compile baseline for Ada RMSNorm.

    Uses torch.compile to generate Triton auto-fusion kernels.
    Supports autograd; both forward and backward are compiled.
    """
    return _adarmsnorm_compiled_fn(x, scale, shift, eps)
