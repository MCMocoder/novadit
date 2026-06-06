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
