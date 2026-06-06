from __future__ import annotations

import torch
import torch.nn.functional as F


def run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=causal,
    )
    return out.transpose(1, 2).contiguous()
