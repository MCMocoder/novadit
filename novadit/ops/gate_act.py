from __future__ import annotations

import torch
import torch.nn.functional as F

from . import novadit_cuda


def geglu_ref(z: torch.Tensor) -> torch.Tensor:
    """Reference GeGLU: y = z_up * GELU(z_gate) using PyTorch tanh-approx GELU."""
    z_up, z_gate = z.chunk(2, dim=-1)
    return z_up * F.gelu(z_gate, approximate="tanh")


def gate_act_ref(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    x_up, x_gate = x.chunk(2, dim=-1)

    if activation == "silu":
        return x_up * F.silu(x_gate)

    if activation == "gelu":
        return geglu_ref(x)

    raise ValueError(f"Unsupported activation: {activation}")


def fused_geglu(z: torch.Tensor) -> torch.Tensor:
    """Fused GeGLU forward: y = z_up * GELU(z_gate) with tanh approximation.

    z: [B, S, 2*I]  --  concatenated up + gate projections.
    Returns y: [B, S, I].
    """
    if novadit_cuda is not None and z.is_cuda:
        return novadit_cuda.fused_geglu_forward(z)
    return geglu_ref(z)


def fused_gate_act(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    if activation not in {"silu", "gelu"}:
        raise ValueError(f"Unsupported activation: {activation}")

    if novadit_cuda is not None and x.is_cuda:
        act_id = 0 if activation == "silu" else 1
        return novadit_cuda.fused_gate_act_forward(x, act_id)
    return gate_act_ref(x, activation)
