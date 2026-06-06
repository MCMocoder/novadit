# ruff: noqa: E402

from __future__ import annotations

import importlib

# Load PyTorch shared libraries before the native extension.
importlib.import_module("torch")

try:
    import novadit_cuda  # type: ignore

    HAS_NOVADIT_CUDA = True
    NOVADIT_CUDA_IMPORT_ERROR = None
except Exception as exc:
    novadit_cuda = None
    HAS_NOVADIT_CUDA = False
    NOVADIT_CUDA_IMPORT_ERROR = exc

from .adaln import (
    adaln_ref,
    fused_adaln,
    fused_adaln_backward_dparam,
    fused_adaln_backward_dx,
)
from .gate_act import fused_gate_act, gate_act_ref
from .qkv_pack import fused_qkv_linear_pack, fused_qkv_pack, qkv_pack_ref
from .residual import fused_out_residual, out_residual_ref

__all__ = [
    "HAS_NOVADIT_CUDA",
    "NOVADIT_CUDA_IMPORT_ERROR",
    "adaln_ref",
    "fused_adaln",
    "fused_adaln_backward_dparam",
    "fused_adaln_backward_dx",
    "fused_gate_act",
    "gate_act_ref",
    "fused_out_residual",
    "fused_qkv_linear_pack",
    "fused_qkv_pack",
    "novadit_cuda",
    "out_residual_ref",
    "qkv_pack_ref",
]
