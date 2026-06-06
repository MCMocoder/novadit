from __future__ import annotations

import torch
import torch.nn.functional as F

from . import novadit_cuda


def gate_act_ref(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    x_up, x_gate = x.chunk(2, dim=-1)

    if activation == "silu":
        return x_up * F.silu(x_gate)

    if activation == "gelu":
        return x_up * F.gelu(x_gate)

    raise ValueError(f"Unsupported activation: {activation}")


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
