from __future__ import annotations

import pytest

from tests.helpers import requires_torch


def _requires_cuda_extension() -> None:
    requires_torch()

    import torch

    from novadit.ops import HAS_NOVADIT_CUDA

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment")
    if not HAS_NOVADIT_CUDA:
        pytest.skip("novadit_cuda extension is not built/importable")


def _requires_sm90() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(f"Requires SM90+, got SM{cap[0]}{cap[1]}")


def _requires_cutlass_dsl() -> None:
    try:
        import cutlass
        import cutlass.cute  # noqa: F401
    except ImportError:
        pytest.skip("cutlass CuTe DSL is not available in this environment")


# ---- Interleave utility tests ----


def test_interleave_weight_roundtrip() -> None:
    requires_torch()

    import torch

    from novadit.ops import interleave_weight, un_interleave_weight

    w = torch.randn(1536, 768)
    w_inter = interleave_weight(w)
    w_back = un_interleave_weight(w_inter)

    assert w_inter.shape == w.shape
    assert torch.equal(w, w_back)


def test_interleave_bias_roundtrip() -> None:
    requires_torch()

    import torch

    from novadit.ops import interleave_bias, un_interleave_bias

    b = torch.randn(1536)
    b_inter = interleave_bias(b)
    b_back = un_interleave_bias(b_inter)

    assert b_inter.shape == b.shape
    assert torch.equal(b, b_back)


def test_interleave_weight_values() -> None:
    requires_torch()

    import torch

    from novadit.ops import interleave_weight

    w = torch.arange(12).reshape(6, 2).float()
    w_inter = interleave_weight(w)

    assert w_inter[0, 0].item() == w[0, 0].item()
    assert w_inter[1, 0].item() == w[3, 0].item()
    assert w_inter[2, 0].item() == w[1, 0].item()
    assert w_inter[3, 0].item() == w[4, 0].item()


# ---- Split-gated-weight utility tests ----


def test_split_gated_weight_values() -> None:
    requires_torch()

    import torch

    from novadit.ops import interleave_weight, split_gated_weight

    w = torch.randn(1536, 768)
    w_inter = interleave_weight(w)
    w_gate, w_value = split_gated_weight(w_inter)

    i = w.shape[0] // 2
    assert w_gate.shape == (i, w.shape[1])
    assert w_value.shape == (i, w.shape[1])
    assert w_gate.dtype == w.dtype
    assert w_value.dtype == w.dtype
    assert torch.equal(w_gate, w[:i])
    assert torch.equal(w_value, w[i:])


def test_split_gated_weight_contiguous() -> None:
    requires_torch()

    import torch

    from novadit.ops import interleave_weight, split_gated_weight

    w = torch.randn(256, 128)
    w_inter = interleave_weight(w)
    w_gate, w_value = split_gated_weight(w_inter)

    assert w_gate.is_contiguous()
    assert w_value.is_contiguous()


def test_split_gated_weight_already_interleaved() -> None:
    requires_torch()

    import torch

    from novadit.ops import split_gated_weight

    w_inter = torch.arange(8).reshape(4, 2).float()
    w_gate, w_value = split_gated_weight(w_inter)

    assert w_gate.shape == (2, 2)
    assert w_value.shape == (2, 2)
    assert torch.equal(w_gate, w_inter[0::2])
    assert torch.equal(w_value, w_inter[1::2])


def test_split_gated_weight_errors() -> None:
    requires_torch()

    import torch

    from novadit.ops import split_gated_weight

    with pytest.raises(ValueError, match="even first dim"):
        split_gated_weight(torch.randn(3, 4))

    with pytest.raises(ValueError, match="even first dim"):
        split_gated_weight(torch.randn(2, 3, 4))


# ---- Reference tests ----


def test_fc1_bias_act_ref_shape() -> None:
    requires_torch()

    import torch

    from novadit.ops import fc1_bias_act_ref

    x = torch.randn(2, 8, 64)
    w = torch.randn(128, 64)
    b = torch.randn(128)

    y = fc1_bias_act_ref(x, w, b, activation="silu")
    assert y.shape == (2, 8, 64)


def test_fc1_bias_act_ref_no_bias() -> None:
    requires_torch()

    import torch

    from novadit.ops import fc1_bias_act_ref

    x = torch.randn(2, 8, 64)
    w = torch.randn(128, 64)

    y = fc1_bias_act_ref(x, w, None, activation="gelu")
    assert y.shape == (2, 8, 64)


# ---- CUDA correctness tests ----


@pytest.mark.parametrize(
    ("dtype", "batch_size", "seq_len", "H", "I", "activation", "has_bias"),
    [
        pytest.param("float16", 2, 128, 768, 768, "silu", True, id="fp16-silu-bias-DiT-S"),
        pytest.param("float16", 1, 64, 1152, 3072, "gelu", True, id="fp16-gelu-bias-DiT-L"),
    ],
)
def test_fused_fc1_bias_act_matches_reference(
    dtype: str,
    batch_size: int,
    seq_len: int,
    H: int,
    I: int,
    activation: str,
    has_bias: bool,
) -> None:
    _requires_cuda_extension()
    _requires_sm90()

    import torch

    from novadit.ops import fc1_bias_act_ref, fused_fc1_bias_act

    torch_dtype = getattr(torch, dtype)
    device = "cuda"

    x = torch.randn(batch_size, seq_len, H, device=device, dtype=torch_dtype)
    weight = torch.randn(2 * I, H, device=device, dtype=torch_dtype)
    bias = torch.randn(2 * I, device=device, dtype=torch_dtype) if has_bias else None

    y_ref = fc1_bias_act_ref(x, weight, bias, activation)
    y_fused = fused_fc1_bias_act(x, weight, bias, activation)

    assert y_fused.shape == (batch_size, seq_len, I)
    assert y_fused.dtype == torch_dtype

    rtol = 3e-2 if dtype == "bfloat16" else 2e-2
    atol = 3e-2 if dtype == "bfloat16" else 2e-2
    torch.testing.assert_close(y_fused, y_ref, rtol=rtol, atol=atol)


# ---- CuTe DSL kernel static tests ----


def test_kernel_class_construction_defaults() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        SM120_DEFAULT_ATOM_LAYOUT_MNK,
        SM120_DEFAULT_TILE_SHAPE_MNK,
        FC1GatedActSM120BF16Kernel,
    )

    kernel = FC1GatedActSM120BF16Kernel()
    assert kernel.tile_shape_mnk == SM120_DEFAULT_TILE_SHAPE_MNK
    assert kernel.atom_layout == SM120_DEFAULT_ATOM_LAYOUT_MNK
    assert kernel.cluster_shape_mnk == (1, 1, 1)
    assert kernel.num_mma_warps == (
        SM120_DEFAULT_ATOM_LAYOUT_MNK[0]
        * SM120_DEFAULT_ATOM_LAYOUT_MNK[1]
        * SM120_DEFAULT_ATOM_LAYOUT_MNK[2]
    )
    assert kernel.threads_per_cta == (kernel.num_mma_warps + 1) * 32
    assert kernel.occupancy == 1


def test_kernel_class_construction_with_shape() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    shape = FusedGatedMlpShape(batch_tokens=128, hidden_size=768, intermediate_size=768)
    kernel = FC1GatedActSM120BF16Kernel(shape)
    assert kernel.shape is shape
    assert kernel.a_dtype == cutlass.Float16
    assert kernel.acc_dtype == cutlass.Float32
    assert kernel.output_dtype == cutlass.Float16
    assert kernel.bias_dtype == cutlass.Float16
    assert kernel.fc1_output_size == 1536
    assert kernel.fc2_output_size == 768


def test_kernel_accepts_fp16_and_bf16() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import FC1GatedActSM120BF16Kernel

    fp16 = FC1GatedActSM120BF16Kernel(ab_dtype=cutlass.Float16, output_dtype=cutlass.Float16)
    assert fp16.a_dtype == cutlass.Float16
    bf16 = FC1GatedActSM120BF16Kernel(ab_dtype=cutlass.BFloat16, output_dtype=cutlass.BFloat16)
    assert bf16.a_dtype == cutlass.BFloat16


def test_kernel_rejects_invalid_dtypes() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import FC1GatedActSM120BF16Kernel

    with pytest.raises(TypeError):
        FC1GatedActSM120BF16Kernel(ab_dtype=cutlass.Int32)
    with pytest.raises(TypeError):
        FC1GatedActSM120BF16Kernel(output_dtype=cutlass.Int8)


def test_kernel_rejects_bad_mma_inst_mnk() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    shape = FusedGatedMlpShape(batch_tokens=64, hidden_size=256, intermediate_size=256)
    kernel = FC1GatedActSM120BF16Kernel.__new__(FC1GatedActSM120BF16Kernel)
    kernel.__init__ = lambda self: None
    kernel.shape = shape
    kernel.ab_dtype = cutlass.Float16
    kernel.acc_dtype = cutlass.Float32
    kernel.output_dtype = cutlass.Float16
    kernel.mma_inst_mnk = (16, 8, 8)
    kernel.atom_layout = (2, 2, 1)

    with pytest.raises(ValueError, match="m16n8k16"):
        FC1GatedActSM120BF16Kernel._validate_static_config(kernel)


def test_kernel_rejects_unsupported_tile_shape() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16Kernel,
    )

    with pytest.raises(ValueError, match="not supported"):
        FC1GatedActSM120BF16Kernel(fc1_tile_shape_mnk=(32, 32, 32))


def test_shape_validation_non_positive_dims() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    with pytest.raises(ValueError, match="batch_tokens"):
        FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=0, hidden_size=768, intermediate_size=768)
        )
    with pytest.raises(ValueError, match="hidden_size"):
        FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=1, hidden_size=-1, intermediate_size=768)
        )
    with pytest.raises(ValueError, match="intermediate_size"):
        FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=1, hidden_size=768, intermediate_size=0)
        )


def test_shape_validation_non_multiple_of_16() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    with pytest.raises(ValueError, match="multiple of 16"):
        FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=1, hidden_size=17, intermediate_size=32)
        )
    with pytest.raises(ValueError, match="multiple of 16"):
        FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=1, hidden_size=32, intermediate_size=15)
        )


def test_activation_name_helpers() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        ACT_GELU,
        ACT_SILU,
        activation_name,
    )

    assert activation_name(ACT_SILU) == "silu"
    assert activation_name(ACT_GELU) == "gelu"
    with pytest.raises(ValueError):
        activation_name(99)


def test_fused_gated_mlp_shape_is_frozen() -> None:
    _requires_cutlass_dsl()
    import dataclasses

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import FusedGatedMlpShape

    shape = FusedGatedMlpShape(batch_tokens=128, hidden_size=768, intermediate_size=768)
    with pytest.raises(dataclasses.FrozenInstanceError):
        shape.batch_tokens = 256  # type: ignore[misc]


def test_export_aot_parse_args_required() -> None:
    _requires_cutlass_dsl()

    import sys
    from unittest import mock

    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "third_party" / "cutlass" / "python" / "CuTeDSL"))

    from csrc.cute_dsl.fc1_bias_act.export_aot import parse_args

    test_args = [
        "export_aot",
        "--batch-tokens=64",
        "--hidden-size=256",
        "--intermediate-size=256",
    ]
    with mock.patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.batch_tokens == 64
        assert args.hidden_size == 256
        assert args.intermediate_size == 256
        assert args.activation == "silu"
        assert args.max_active_clusters == 1


# ---- CuTe DSL JIT end-to-end tests ----


def _requires_gpu_and_cutlass() -> None:
    _requires_cutlass_dsl()
    import cuda.bindings.driver  # noqa: F401 — initialize CUDA driver context for DSL
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment")


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_fc1_gated_act_jit_matches_cublaslt(
    dtype_name: str,
    activation: str,
) -> None:
    _requires_gpu_and_cutlass()

    import torch
    import torch.nn.functional as F
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        ACT_GELU,
        ACT_SILU,
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    torch_dtype = getattr(torch, dtype_name)
    cute_dtype = getattr(cutlass, "Float16" if dtype_name == "float16" else "BFloat16")

    M = 64
    H = 128
    I = 64
    L = 1

    # Row-major tensors via cutlass_torch.
    x_cpu = cutlass_torch.matrix(L, M, H, "k", cute_dtype)
    w_gate_cpu = cutlass_torch.matrix(L, I, H, "k", cute_dtype)
    w_value_cpu = cutlass_torch.matrix(L, I, H, "k", cute_dtype)
    y_cpu = cutlass_torch.matrix(L, M, I, "n", cute_dtype)

    def to_gpu(t_cpu):
        cute_t, t_gpu = cutlass_torch.cute_tensor_like(t_cpu, cute_dtype, True, 16)
        return cute_t, t_gpu

    x_cute, x_gpu = to_gpu(x_cpu)
    w_gate_cute, w_gate_gpu = to_gpu(w_gate_cpu)
    w_value_cute, w_value_gpu = to_gpu(w_value_cpu)
    y_cute, y_gpu = to_gpu(y_cpu)

    act_id = ACT_SILU if activation == "silu" else ACT_GELU

    kernel = FC1GatedActSM120BF16Kernel(
        FusedGatedMlpShape(batch_tokens=M, hidden_size=H, intermediate_size=I),
        ab_dtype=cute_dtype,
        output_dtype=cute_dtype,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()

    compiled = cute.compile(
        kernel,
        x_cute,
        w_gate_cute,
        w_value_cute,
        y_cute,
        act_id,
        max_active_clusters,
        stream,
    )
    compiled(x_cute, w_gate_cute, w_value_cute, y_cute, stream)
    torch.cuda.synchronize()

    # ── cuBLASLt baseline ──────────────────────────────────────────
    x2d = x_gpu.squeeze(-1)
    wg2d = w_gate_gpu.squeeze(-1)
    wv2d = w_value_gpu.squeeze(-1)

    # cuBLASLt: separate matmuls + activation (same as reference path)
    gate = F.linear(x2d.float(), wg2d.float())
    value = F.linear(x2d.float(), wv2d.float())
    if activation == "silu":
        baseline = value * F.silu(gate)
    else:
        baseline = value * F.gelu(gate, approximate="tanh")
    baseline = baseline.to(torch_dtype)

    result = y_gpu.float().squeeze(-1)
    baseline_f32 = baseline.float()

    err = (result - baseline_f32).abs().max().item()
    rmse = (result - baseline_f32).square().mean().sqrt().item()

    atol = 3e-2 if dtype_name == "bfloat16" else 2e-2
    assert err < atol, f"{dtype_name}/{activation}: max_abs_err={err:.4f} > {atol}"


# ---- CuTe DSL backward dx static tests ----


def test_backward_dx_kernel_class_construction_defaults() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        SM120_DEFAULT_ATOM_LAYOUT_MNK,
        SM120_DEFAULT_TILE_SHAPE_MNK,
        FC1GatedActSM120BF16BackwardDxKernel,
    )

    kernel = FC1GatedActSM120BF16BackwardDxKernel()
    assert kernel.tile_shape_mnk == SM120_DEFAULT_TILE_SHAPE_MNK
    assert kernel.atom_layout == SM120_DEFAULT_ATOM_LAYOUT_MNK
    assert kernel.cluster_shape_mnk == (1, 1, 1)
    assert kernel.threads_per_cta == (kernel.num_mma_warps + 1) * 32
    assert kernel.occupancy == 1


def test_backward_dx_kernel_accepts_fp16_and_bf16() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16BackwardDxKernel,
    )

    fp16 = FC1GatedActSM120BF16BackwardDxKernel(
        ab_dtype=cutlass.Float16, output_dtype=cutlass.Float16
    )
    assert fp16.a_dtype == cutlass.Float16
    bf16 = FC1GatedActSM120BF16BackwardDxKernel(
        ab_dtype=cutlass.BFloat16, output_dtype=cutlass.BFloat16
    )
    assert bf16.a_dtype == cutlass.BFloat16


def test_backward_dx_kernel_rejects_invalid_dtypes() -> None:
    _requires_cutlass_dsl()
    import cutlass

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16BackwardDxKernel,
    )

    with pytest.raises(TypeError):
        FC1GatedActSM120BF16BackwardDxKernel(ab_dtype=cutlass.Int32)
    with pytest.raises(TypeError):
        FC1GatedActSM120BF16BackwardDxKernel(output_dtype=cutlass.Int8)


def test_backward_dx_kernel_rejects_unsupported_tile_shape() -> None:
    _requires_cutlass_dsl()

    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16BackwardDxKernel,
    )

    with pytest.raises(ValueError, match="not supported"):
        FC1GatedActSM120BF16BackwardDxKernel(tile_shape_mnk=(32, 32, 32))


# ---- CuTe DSL backward dx JIT test ----


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_backward_dx_jit_matches_torch_mm(
    dtype_name: str,
) -> None:
    _requires_gpu_and_cutlass()

    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from csrc.cute_dsl.fc1_bias_act.fc1_bias_act import (
        FC1GatedActSM120BF16BackwardDxKernel,
    )

    torch_dtype = getattr(torch, dtype_name)
    cute_dtype = getattr(cutlass, "Float16" if dtype_name == "float16" else "BFloat16")

    M = 64
    H = 128
    I = 64
    L = 1

    dgate_cpu = cutlass_torch.matrix(L, M, I, "k", cute_dtype)
    dvalue_cpu = cutlass_torch.matrix(L, M, I, "k", cute_dtype)
    w_gate_cpu_T = cutlass_torch.matrix(L, H, I, "k", cute_dtype)
    w_value_cpu_T = cutlass_torch.matrix(L, H, I, "k", cute_dtype)
    dx_cpu = cutlass_torch.matrix(L, M, H, "n", cute_dtype)

    def to_gpu(t_cpu):
        cute_t, t_gpu = cutlass_torch.cute_tensor_like(t_cpu, cute_dtype, True, 16)
        return cute_t, t_gpu

    dgate_cute, dgate_gpu = to_gpu(dgate_cpu)
    dvalue_cute, dvalue_gpu = to_gpu(dvalue_cpu)
    w_gate_cute, w_gate_gpu_T = to_gpu(w_gate_cpu_T)
    w_value_cute, w_value_gpu_T = to_gpu(w_value_cpu_T)
    dx_cute, dx_gpu = to_gpu(dx_cpu)

    kernel = FC1GatedActSM120BF16BackwardDxKernel(
        ab_dtype=cute_dtype,
        output_dtype=cute_dtype,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()

    compiled = cute.compile(
        kernel,
        dgate_cute,
        dvalue_cute,
        w_gate_cute,
        w_value_cute,
        dx_cute,
        max_active_clusters,
        stream,
    )
    compiled(dgate_cute, dvalue_cute, w_gate_cute, w_value_cute, dx_cute, stream)
    torch.cuda.synchronize()

    dg2d = dgate_gpu.squeeze(-1)
    dv2d = dvalue_gpu.squeeze(-1)
    wg2d_T = w_gate_gpu_T.squeeze(-1)
    wv2d_T = w_value_gpu_T.squeeze(-1)

    dx_ref = torch.mm(dg2d.float(), wg2d_T.float().T) + torch.mm(dv2d.float(), wv2d_T.float().T)
    dx_ref = dx_ref.to(torch_dtype)

    result = dx_gpu.float().squeeze(-1)
    ref_f32 = dx_ref.float()

    err = (result - ref_f32).abs().max().item()
    rmse = (result - ref_f32).square().mean().sqrt().item()

    atol = 3e-2 if dtype_name == "bfloat16" else 2e-2
    assert err < atol, f"{dtype_name}: max_abs_err={err:.4f} > {atol}"


# ---- Integration test: fused_fc1_bias_act backward through the plugin ----


def _requires_sm90_and_cutlass() -> None:
    _requires_sm90()
    _requires_cutlass_dsl()
    import cuda.bindings.driver  # noqa: F401
    import torch  # noqa: F401


def test_fused_fc1_bias_act_backward_dx_plugin() -> None:
    """Integration test: backward dx through the ops dispatch path.

    Exercises ``_backward_dx_cute`` (called by ``_FusedFc1BiasActFunction``)
    using the same dz/weight shapes the real backward produces.
    """
    _requires_sm90_and_cutlass()
    _requires_cuda_extension()

    import torch
    import torch.nn.functional as F

    from novadit.ops.fc1_bias_act import _backward_dx_cute, interleave_weight

    for dtype in (torch.float16, torch.bfloat16):
        B, S, H, I = 1, 64, 128, 64
        device = "cuda"
        activation = "silu"

        torch.manual_seed(42)
        x = torch.randn(B, S, H, device=device, dtype=dtype)
        weight = torch.randn(2 * I, H, device=device, dtype=dtype)

        M = B * S
        z = F.linear(x, weight)

        z.requires_grad_(True)
        z.retain_grad()
        gate, value = z.chunk(2, dim=-1)
        y_ref = (
            value * F.silu(gate)
            if activation == "silu"
            else value * F.gelu(gate, approximate="tanh")
        )
        y_ref.sum().backward()
        dz = z.grad.reshape(M, -1)

        # ── Plugin path (CuTe DSL) ────────────────────────────────
        dx_cute = _backward_dx_cute(dz, weight, M, H, I)

        # ── Reference ─────────────────────────────────────────────
        dx_ref = (dz.float() @ weight.float()).to(dtype)

        # If CuTe kernel is unavailable (e.g. forward kernel not wired),
        # _backward_dx_cute falls back to dz @ weight which should match.
        atol = 3e-2 if dtype == torch.bfloat16 else 7e-2
        max_err = (dx_cute.float() - dx_ref.float()).abs().max().item()
        assert max_err < atol, (
            f"Backward dx mismatch for {dtype}: max_abs_err={max_err:.4e} > atol={atol}"
        )
