from __future__ import annotations

import torch

from . import novadit_cuda


class _FusedAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        y, scale_contig, mean, invstd = novadit_cuda.fused_adaln_forward(
            x, scale, shift, eps
        )
        ctx.save_for_backward(x, scale_contig, mean, invstd)
        ctx.scale_shape = scale.shape
        ctx.shift_shape = shift.shape
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        x, scale, mean, invstd = ctx.saved_tensors

        dx = None
        if ctx.needs_input_grad[0]:
            dx = novadit_cuda.fused_adaln_backward_dx(dy, x, scale, mean, invstd)

        dscale = None
        dshift = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dscale_cuda, dshift_cuda = novadit_cuda.fused_adaln_backward_dparam(
                dy, x, mean, invstd
            )
            if ctx.needs_input_grad[1]:
                dscale = dscale_cuda.reshape(ctx.scale_shape)
            if ctx.needs_input_grad[2]:
                dshift = dshift_cuda.reshape(ctx.shift_shape)

        return dx, dscale, dshift, None


def _broadcast_modulation(x: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    while tensor.dim() < x.dim():
        tensor = tensor.unsqueeze(1)
    return tensor


def adaln_ref(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    xhat = (x.float() - mean) * torch.rsqrt(var + eps)

    scale = _broadcast_modulation(x, scale).float()
    shift = _broadcast_modulation(x, shift).float()
    y = xhat * (1.0 + scale) + shift
    return y.to(dtype=x.dtype)


def fused_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if novadit_cuda is not None and x.is_cuda:
        return _FusedAdaLNFunction.apply(x, scale, shift, eps)
    return adaln_ref(x, scale, shift, eps)


def fused_adaln_backward_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
) -> torch.Tensor:
    if novadit_cuda is None:
        raise RuntimeError("novadit_cuda extension is not available")
    return novadit_cuda.fused_adaln_backward_dx(dy, x, scale, mean, invstd)


def fused_adaln_backward_dparam(
    dy: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if novadit_cuda is None:
        raise RuntimeError("novadit_cuda extension is not available")
    return novadit_cuda.fused_adaln_backward_dparam(dy, x, mean, invstd)
