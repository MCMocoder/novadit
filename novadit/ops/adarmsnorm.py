from __future__ import annotations

import torch

from . import novadit_cuda


class _FusedAdaRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        y, scale_contig, inv_rms = novadit_cuda.fused_adarmsnorm_forward(
            x, scale, shift, eps
        )
        ctx.save_for_backward(x, scale_contig, inv_rms)
        ctx.scale_shape = scale.shape
        ctx.shift_shape = shift.shape
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        x, scale, inv_rms = ctx.saved_tensors

        dx = None
        if ctx.needs_input_grad[0]:
            dx = novadit_cuda.fused_adarmsnorm_backward_dx(dy, x, scale, inv_rms)

        dscale = None
        dshift = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dscale_cuda, dshift_cuda = novadit_cuda.fused_adarmsnorm_backward_dparam(
                dy, x, inv_rms
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


def adarmsnorm_ref(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference AdaRMSNorm: y = (x / rms(x)) * (1 + scale) + shift."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = x.float() / rms

    scale = _broadcast_modulation(x, scale).float()
    shift = _broadcast_modulation(x, shift).float()
    y = xhat * (1.0 + scale) + shift
    return y.to(dtype=x.dtype)


def fused_adarmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused AdaRMSNorm forward with autograd support."""
    if novadit_cuda is not None and x.is_cuda:
        return _FusedAdaRMSNormFunction.apply(x, scale, shift, eps)
    return adarmsnorm_ref(x, scale, shift, eps)


@torch.no_grad()
def fused_adarmsnorm_inference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Inference-only fused AdaRMSNorm forward (returns y only, no stats)."""
    if novadit_cuda is not None and x.is_cuda:
        return novadit_cuda.fused_adarmsnorm_forward_inference(x, scale, shift, eps)
    return adarmsnorm_ref(x, scale, shift, eps)
