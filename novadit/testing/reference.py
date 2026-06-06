from __future__ import annotations

import torch


def make_attention_inputs(
    batch_size: int = 2,
    seq_len: int = 16,
    hidden_size: int = 64,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)


def make_adaln_inputs(
    batch_size: int = 2,
    seq_len: int = 16,
    hidden_size: int = 64,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    scale = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    shift = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    return x, scale, shift
