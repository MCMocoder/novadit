from __future__ import annotations

import torch

from .backends import AUTO_BACKEND, SUPPORTED_BACKENDS
from .flash_attn_compat import has_flash_attn, import_flash_attn
from .sdpa import run_sdpa


def get_attention_backend(
    backend: str = AUTO_BACKEND,
    attention_mask: torch.Tensor | None = None,
) -> str:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported attention backend: {backend}")

    if backend == AUTO_BACKEND:
        if attention_mask is None and has_flash_attn():
            return "fa2"
        return "sdpa"

    if backend.startswith("fa") and attention_mask is not None:
        return "sdpa"

    if backend.startswith("fa") and not has_flash_attn():
        return "sdpa"

    return backend


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    attention_mask: torch.Tensor | None = None,
    backend: str = AUTO_BACKEND,
) -> torch.Tensor:
    selected = get_attention_backend(backend=backend, attention_mask=attention_mask)

    if selected in {"sdpa", "cudnn_sdpa", "math"}:
        return run_sdpa(q, k, v, dropout_p=dropout_p, causal=causal, attention_mask=attention_mask)

    if selected.startswith("fa"):
        # FlashAttention integration will be filled in during kernel/runtime work.
        return run_sdpa(q, k, v, dropout_p=dropout_p, causal=causal, attention_mask=attention_mask)

    raise RuntimeError(f"Unexpected backend selection: {selected}")


def run_attention_qkvpacked(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    attention_mask: torch.Tensor | None = None,
    backend: str = AUTO_BACKEND,
) -> torch.Tensor:
    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError("qkv must have shape [B, S, 3, num_heads, head_dim]")

    selected = get_attention_backend(
        backend=backend,
        attention_mask=attention_mask,
    )
    flash_supported = qkv.is_cuda and qkv.dtype in {
        torch.float16,
        torch.bfloat16,
    }

    if selected.startswith("fa") and flash_supported:
        flash_attn = import_flash_attn()
        qkvpacked_func = getattr(
            flash_attn,
            "flash_attn_qkvpacked_func",
            None,
        )
        if qkvpacked_func is not None:
            return qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                causal=causal,
            )

    q, k, v = qkv.unbind(dim=2)
    return run_sdpa(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        attention_mask=attention_mask,
    )
