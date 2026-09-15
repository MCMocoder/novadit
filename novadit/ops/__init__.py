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
    fused_adaln_inference,
    fused_adaln_backward_dparam,
    fused_adaln_backward_dx,
)
from .adarmsnorm import (
    adarmsnorm_ref,
    fused_adarmsnorm,
    fused_adarmsnorm_inference,
)
from .gate_act import fused_geglu, fused_gate_act, gate_act_ref, geglu_ref
from .fc1_bias_act import (
    fc1_bias_act_ref,
    fused_fc1_bias_act,
    interleave_bias,
    interleave_weight,
    split_gated_weight,
    un_interleave_bias,
    un_interleave_weight,
)
from .qkv_pack import fused_qkv_linear_pack, fused_qkv_pack, qkv_pack_ref
from .norm_compile import adaln_compile, adarmsnorm_compile
from .residual import fused_out_residual, out_residual_ref

__all__ = [
    "HAS_NOVADIT_CUDA",
    "NOVADIT_CUDA_IMPORT_ERROR",
    "adarmsnorm_ref",
    "adaln_ref",
    "fused_adaln",
    "fused_adaln_inference",
    "fused_adaln_backward_dparam",
    "fused_adaln_backward_dx",
    "fused_adarmsnorm",
    "fused_adarmsnorm_inference",
    "fused_geglu",
    "fused_gate_act",
    "gate_act_ref",
    "geglu_ref",
    "fc1_bias_act_ref",
    "fused_fc1_bias_act",
    "interleave_weight",
    "un_interleave_weight",
    "interleave_bias",
    "un_interleave_bias",
    "split_gated_weight",
    "fused_out_residual",
    "fused_qkv_linear_pack",
    "fused_qkv_pack",
    "adaln_compile",
    "adarmsnorm_compile",
    "novadit_cuda",
    "out_residual_ref",
    "qkv_pack_ref",
]
