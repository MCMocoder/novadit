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


# ---- GeGLU-specific tests ----

def test_geglu_ref_shape_and_dtype() -> None:
    requires_torch()

    import torch

    from novadit.ops import geglu_ref

    z = torch.randn(2, 8, 128, dtype=torch.float32)
    y = geglu_ref(z)

    assert y.shape == (2, 8, 64)
    assert y.dtype == z.dtype


def test_fused_geglu_matches_reference() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_geglu, geglu_ref

    z = torch.randn(2, 8, 64, dtype=torch.float32)
    y_ref = geglu_ref(z)
    y = fused_geglu(z)

    torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("dtype", "batch_size", "seq_len", "inter_size", "rtol", "atol"),
    [
        pytest.param("float16", 2, 4, 768, 2e-2, 2e-2, id="fp16-aligned"),
        pytest.param("bfloat16", 2, 4, 768, 2e-2, 2e-2, id="bf16-aligned"),
        pytest.param("float32", 2, 4, 768, 1e-4, 1e-4, id="fp32-aligned"),
        # inter_size=385 forces dynamic path (not a known aligned size)
        pytest.param("float16", 2, 4, 385, 2e-2, 2e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", 2, 4, 385, 3e-2, 3e-2, id="bf16-dynamic"),
        pytest.param("float32", 2, 4, 385, 1e-4, 1e-4, id="fp32-dynamic"),
    ],
)
def test_fused_geglu_matches_reference_on_cuda(
    dtype: str,
    batch_size: int,
    seq_len: int,
    inter_size: int,
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_geglu, geglu_ref

    torch_dtype = getattr(torch, dtype)
    device = "cuda"

    z = torch.randn(
        batch_size, seq_len, 2 * inter_size, device=device, dtype=torch_dtype
    )

    y_ref = geglu_ref(z)
    y = fused_geglu(z)

    assert y.shape == (batch_size, seq_len, inter_size)
    assert y.dtype == z.dtype
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)


# ---- Gate-act dispatch tests ----

def test_gate_act_shape() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_gate_act

    x = torch.randn(2, 8, 64)
    y = fused_gate_act(x, activation="silu")
    assert y.shape == (2, 8, 32)


def test_gate_act_gelu_matches_geglu() -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_gate_act, fused_geglu

    z = torch.randn(2, 4, 128, dtype=torch.float32)

    y_geglu = fused_geglu(z)
    y_gate = fused_gate_act(z, activation="gelu")

    torch.testing.assert_close(y_gate, y_geglu)


@pytest.mark.parametrize(
    ("dtype", "inter_size", "rtol", "atol"),
    [
        pytest.param("float16", 768, 2e-2, 2e-2, id="fp16-aligned"),
        pytest.param("bfloat16", 768, 2e-2, 2e-2, id="bf16-aligned"),
        pytest.param("float32", 768, 1e-4, 1e-4, id="fp32-aligned"),
        pytest.param("float16", 385, 2e-2, 2e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", 385, 3e-2, 3e-2, id="bf16-dynamic"),
        pytest.param("float32", 385, 1e-4, 1e-4, id="fp32-dynamic"),
    ],
)
def test_fused_gate_act_matches_reference_on_cuda(
    dtype: str,
    inter_size: int,
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_gate_act, gate_act_ref

    torch_dtype = getattr(torch, dtype)
    device = "cuda"

    x = torch.randn(2, 3, 2 * inter_size, device=device, dtype=torch_dtype)

    y_ref_silu = gate_act_ref(x, activation="silu")
    y_silu = fused_gate_act(x, activation="silu")
    torch.testing.assert_close(y_silu, y_ref_silu, rtol=rtol, atol=atol)

    y_ref_gelu = gate_act_ref(x, activation="gelu")
    y_gelu = fused_gate_act(x, activation="gelu")
    torch.testing.assert_close(y_gelu, y_ref_gelu, rtol=rtol, atol=atol)

# ---- Backward tests ----

def _ref_geglu_backward(
    dy: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Reference backward for GeGLU: dz = [dy*gelu(z_gate), dy*z_up*gelu'(z_gate)]."""
    z_up, z_gate = z.chunk(2, dim=-1)
    # Forward GELU (tanh approx)
    gelu_gate = F.gelu(z_gate, approximate="tanh")
    # GELU backward via autograd
    z_up_ref = z_up.detach().clone().requires_grad_(True)
    z_gate_ref = z_gate.detach().clone().requires_grad_(True)
    y_ref = z_up_ref * F.gelu(z_gate_ref, approximate="tanh")
    y_ref.backward(dy)
    return torch.cat([z_up_ref.grad, z_gate_ref.grad], dim=-1)


@pytest.mark.parametrize(
    ("dtype", "batch_size", "seq_len", "inter_size", "rtol", "atol"),
    [
        pytest.param("float16", 2, 3, 768, 3e-2, 3e-2, id="fp16-aligned"),
        pytest.param("bfloat16", 2, 3, 768, 4e-2, 4e-2, id="bf16-aligned"),
        pytest.param("float32", 2, 3, 768, 2e-4, 2e-4, id="fp32-aligned"),
        pytest.param("float16", 2, 3, 385, 4e-2, 4e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", 2, 3, 385, 5e-2, 5e-2, id="bf16-dynamic"),
        pytest.param("float32", 2, 3, 385, 3e-4, 3e-4, id="fp32-dynamic"),
    ],
)
def test_fused_geglu_backward_matches_reference_on_cuda(
    dtype: str,
    batch_size: int,
    seq_len: int,
    inter_size: int,
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch
    import torch.nn.functional as F

    from novadit.ops import fused_geglu, novadit_cuda

    torch_dtype = getattr(torch, dtype)
    device = "cuda"

    z = torch.randn(
        batch_size, seq_len, 2 * inter_size, device=device, dtype=torch_dtype,
        requires_grad=True,
    )
    dy = torch.randn(
        batch_size, seq_len, inter_size, device=device, dtype=torch_dtype,
    )

    y = fused_geglu(z)
    y.backward(dy)

    dz_expected = _ref_geglu_backward(dy.float(), z.float()).to(torch_dtype)

    assert z.grad is not None
    torch.testing.assert_close(z.grad, dz_expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_fused_gate_act_backward_matches_reference_on_cuda(
    activation: str,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_gate_act, gate_act_ref

    device = "cuda"

    x = torch.randn(2, 3, 2 * 256, device=device, dtype=torch.float32,
                    requires_grad=True)
    dy = torch.randn(2, 3, 256, device=device, dtype=torch.float32)

    # Fused
    y_fused = fused_gate_act(x, activation=activation)
    y_fused.backward(dy)
    dz_fused = x.grad.clone()

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = gate_act_ref(x_ref, activation=activation)
    y_ref.backward(dy)
    dz_ref = x_ref.grad

    assert dz_fused is not None
    torch.testing.assert_close(dz_fused, dz_ref, rtol=5e-4, atol=5e-4)


def test_fused_geglu_autograd_consistency() -> None:
    """Verify fused_geglu + backward produces same grads as geglu_ref + autograd."""
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_geglu, geglu_ref

    device = "cuda"

    z_fused = torch.randn(2, 3, 2 * 256, device=device, dtype=torch.float32,
                          requires_grad=True)
    z_ref = z_fused.detach().clone().requires_grad_(True)
    dy = torch.randn(2, 3, 256, device=device, dtype=torch.float32)

    y_fused = fused_geglu(z_fused)
    y_fused.backward(dy)

    y_ref = geglu_ref(z_ref)
    y_ref.backward(dy)

    torch.testing.assert_close(y_fused, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(z_fused.grad, z_ref.grad, rtol=1e-4, atol=1e-4)

