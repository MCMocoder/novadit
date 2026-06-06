from __future__ import annotations

import torch
from torch import nn

from novadit.modules.dit_attention import NovaDiTAttention
from novadit.modules.dit_mlp import NovaDiTMLP
from novadit.ops import adaln_ref, fused_adaln


class NovaDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.0,
        attention_backend: str = "auto",
        use_fused_adaln: bool = True,
        use_fused_qkv_pack: bool = True,
        use_fused_out_residual: bool = True,
        use_fused_gate_act: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.use_fused_adaln = use_fused_adaln
        self.attn = NovaDiTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=dropout_p,
            attention_backend=attention_backend,
            use_fused_qkv_pack=use_fused_qkv_pack,
            use_fused_out_residual=use_fused_out_residual,
            add_residual=False,
            device=device,
            dtype=dtype,
        )
        self.mlp = NovaDiTMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_p=dropout_p,
            use_fused_gate_act=use_fused_gate_act,
            add_residual=False,
            device=device,
            dtype=dtype,
        )

    def _adaln(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_fused_adaln:
            return fused_adaln(x, scale, shift)
        return adaln_ref(x, scale, shift)

    def forward(
        self,
        x: torch.Tensor,
        attn_scale: torch.Tensor,
        attn_shift: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_shift: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self._adaln(x, attn_scale, attn_shift), attention_mask=attention_mask)
        x = x + self.mlp(self._adaln(x, mlp_scale, mlp_shift))
        return x
