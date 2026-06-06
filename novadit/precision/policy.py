from __future__ import annotations

import torch


def validate_training_dtype(dtype: torch.dtype) -> None:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"Unsupported training dtype: {dtype}")


def autocast_dtype_for_training(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32
