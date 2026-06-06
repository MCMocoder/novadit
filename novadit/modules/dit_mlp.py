from __future__ import annotations

import torch
from torch import nn

from novadit.ops import fused_gate_act, gate_act_ref


class NovaDiTMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = True,
        dropout_p: float = 0.0,
        use_fused_gate_act: bool = True,
        add_residual: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_p = dropout_p
        self.use_fused_gate_act = use_fused_gate_act
        self.add_residual = add_residual

        self.fc1 = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)

        if self.use_fused_gate_act:
            x = fused_gate_act(x, activation=self.activation)
        else:
            x = gate_act_ref(x, activation=self.activation)

        x = self.fc2(x)
        if self.dropout_p > 0:
            x = self.dropout(x)

        if self.add_residual:
            return residual + x
        return x
