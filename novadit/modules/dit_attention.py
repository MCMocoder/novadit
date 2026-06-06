from __future__ import annotations

import torch
from torch import nn

from novadit.attention import run_attention_qkvpacked
from novadit.ops import fused_out_residual


class NovaDiTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        qkv_bias: bool = True,
        out_bias: bool = True,
        dropout_p: float = 0.0,
        causal: bool = False,
        attention_backend: str = "auto",
        use_fused_qkv_pack: bool = True,
        use_fused_out_residual: bool = True,
        add_residual: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout_p = dropout_p
        self.causal = causal
        self.attention_backend = attention_backend
        self.use_fused_qkv_pack = use_fused_qkv_pack
        self.use_fused_out_residual = use_fused_out_residual
        self.add_residual = add_residual

        if self.num_heads * self.head_dim != hidden_size:
            raise ValueError("hidden_size must equal num_heads * head_dim")

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=out_bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        attn_out = run_attention_qkvpacked(
            qkv,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=self.causal,
            attention_mask=attention_mask,
            backend=self.attention_backend,
        )

        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_size)
        projected = self.out_proj(attn_out)

        if not self.add_residual:
            return self.dropout(projected) if self.dropout_p > 0 else projected

        if self.dropout_p == 0.0 and self.use_fused_out_residual:
            return fused_out_residual(projected, residual, None)

        return residual + self.dropout(projected)
