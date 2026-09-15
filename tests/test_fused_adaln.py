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


@pytest.mark.parametrize("mod_shape", [(2, 16), (2, 1, 16)])
def test_adaln_ref_forward_shape_and_dtype(mod_shape: tuple[int, ...]) -> None:
    requires_torch()

    import torch

    from novadit.ops import fused_adaln

    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    scale = torch.randn(*mod_shape, dtype=torch.bfloat16)
    shift = torch.randn(*mod_shape, dtype=torch.bfloat16)

    y = fused_adaln(x, scale, shift)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_adaln_ref_matches_manual_formula() -> None:
    requires_torch()

    import torch

    from novadit.ops import adaln_ref, fused_adaln

    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    shift = torch.randn(2, 1, 8, dtype=torch.bfloat16)

    ref = adaln_ref(x, scale, shift, eps=1e-5)
    out = fused_adaln(x, scale, shift, eps=1e-5)

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("dtype", "shape", "rtol", "atol"),
    [
        pytest.param("float16", (2, 3, 768), 1e-2, 1e-2, id="fp16-aligned"),
        pytest.param("bfloat16", (2, 3, 768), 2e-2, 2e-2, id="bf16-aligned"),
        # H=770 forces the dynamic path, and row 1+ becomes misaligned for fp16/bf16.
        pytest.param("float16", (2, 3, 770), 2e-2, 2e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", (2, 3, 770), 3e-2, 3e-2, id="bf16-dynamic"),
    ],
)
def test_fused_adaln_forward_matches_reference_on_cuda(
    dtype: str,
    shape: tuple[int, int, int],
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import adaln_ref, fused_adaln

    torch_dtype = getattr(torch, dtype)
    batch_size, seq_len, hidden_size = shape
    device = "cuda"

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch_dtype)
    scale = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    shift = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)

    ref = adaln_ref(x, scale, shift, eps=1e-5)
    out = fused_adaln(x, scale, shift, eps=1e-5)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "shape", "rtol", "atol"),
    [
        pytest.param("float16", (2, 3, 768), 1e-2, 1e-2, id="fp16-aligned"),
        pytest.param("bfloat16", (2, 3, 768), 2e-2, 2e-2, id="bf16-aligned"),
        pytest.param("float16", (2, 3, 770), 2e-2, 2e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", (2, 3, 770), 3e-2, 3e-2, id="bf16-dynamic"),
    ],
)
def test_fused_adaln_inference_matches_reference_on_cuda(
    dtype: str,
    shape: tuple[int, int, int],
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import adaln_ref, fused_adaln_inference

    torch_dtype = getattr(torch, dtype)
    batch_size, seq_len, hidden_size = shape
    device = "cuda"

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch_dtype)
    scale = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    shift = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)

    ref = adaln_ref(x, scale, shift, eps=1e-5)
    out = fused_adaln_inference(x, scale, shift, eps=1e-5)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_fused_adaln_inference_matches_fused_adaln_on_cuda() -> None:
    """Inference path produces identical output to the training forward path."""
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_adaln, fused_adaln_inference

    x = torch.randn(2, 4, 768, device="cuda", dtype=torch.float16)
    scale = torch.randn(2, 768, device="cuda", dtype=torch.float16)
    shift = torch.randn(2, 768, device="cuda", dtype=torch.float16)

    y_train = fused_adaln(x, scale, shift, eps=1e-5)
    y_infer = fused_adaln_inference(x, scale, shift, eps=1e-5)

    torch.testing.assert_close(y_infer, y_train)


@pytest.mark.parametrize(
    ("dtype", "hidden_size", "rtol", "atol"),
    [
        pytest.param("float16", 768, 2e-2, 2e-2, id="fp16-aligned"),
        pytest.param("bfloat16", 768, 4e-2, 4e-2, id="bf16-aligned"),
        pytest.param("float16", 770, 3e-2, 3e-2, id="fp16-dynamic"),
        pytest.param("bfloat16", 770, 5e-2, 5e-2, id="bf16-dynamic"),
    ],
)
def test_fused_adaln_backward_dx_matches_formula_on_cuda(
    dtype: str,
    hidden_size: int,
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_adaln_backward_dx, novadit_cuda

    torch_dtype = getattr(torch, dtype)
    batch_size, seq_len = 2, 3
    x = torch.randn(
        batch_size, seq_len, hidden_size, device="cuda", dtype=torch_dtype
    )
    scale = torch.randn(
        batch_size, hidden_size, device="cuda", dtype=torch_dtype
    )
    shift = torch.randn_like(scale)
    dy = torch.randn_like(x)

    y, saved_scale, mean, invstd = novadit_cuda.fused_adaln_forward(
        x, scale, shift, 1e-5
    )

    ref_mean = x.float().mean(dim=-1)
    ref_var = x.float().var(dim=-1, unbiased=False)
    ref_invstd = torch.rsqrt(ref_var + 1e-5)
    xhat = (x.float() - ref_mean.unsqueeze(-1)) * ref_invstd.unsqueeze(-1)
    dxhat = dy.float() * (1.0 + scale.float().unsqueeze(1))
    expected = ref_invstd.unsqueeze(-1) * (
        dxhat
        - dxhat.mean(dim=-1, keepdim=True)
        - xhat * (dxhat * xhat).mean(dim=-1, keepdim=True)
    )

    dx = fused_adaln_backward_dx(dy, x, saved_scale, mean, invstd)

    assert y.shape == x.shape
    assert saved_scale.shape == scale.shape
    assert mean.shape == (batch_size, seq_len)
    assert invstd.shape == (batch_size, seq_len)
    assert mean.dtype == torch.float32
    assert invstd.dtype == torch.float32
    torch.testing.assert_close(mean, ref_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(invstd, ref_invstd, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dx, expected.to(torch_dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "hidden_size", "rtol", "atol"),
    [
        pytest.param("float16", 768, 2e-2, 2e-2, id="fp16-aligned"),
        pytest.param("bfloat16", 768, 4e-2, 4e-2, id="bf16-aligned"),
        pytest.param("float16", 770, 2e-2, 2e-2, id="fp16-tail"),
        pytest.param("bfloat16", 770, 4e-2, 4e-2, id="bf16-tail"),
    ],
)
def test_fused_adaln_backward_dparam_matches_formula_on_cuda(
    dtype: str,
    hidden_size: int,
    rtol: float,
    atol: float,
) -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import fused_adaln_backward_dparam, novadit_cuda

    torch_dtype = getattr(torch, dtype)
    batch_size, seq_len = 2, 257
    x = torch.randn(
        batch_size, seq_len, hidden_size, device="cuda", dtype=torch_dtype
    )
    scale = torch.randn(
        batch_size, hidden_size, device="cuda", dtype=torch_dtype
    )
    shift = torch.randn_like(scale)
    dy = torch.randn_like(x)

    _, _, mean, invstd = novadit_cuda.fused_adaln_forward(
        x, scale, shift, 1e-5
    )
    xhat = (x.float() - mean.unsqueeze(-1)) * invstd.unsqueeze(-1)
    expected_dscale = (dy.float() * xhat).sum(dim=1)
    expected_dshift = dy.float().sum(dim=1)

    dscale, dshift = fused_adaln_backward_dparam(dy, x, mean, invstd)

    assert dscale.shape == (batch_size, hidden_size)
    assert dshift.shape == (batch_size, hidden_size)
    assert dscale.dtype == torch.float32
    assert dshift.dtype == torch.float32
    torch.testing.assert_close(
        dscale, expected_dscale, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        dshift, expected_dshift, rtol=rtol, atol=atol
    )


def test_fused_adaln_autograd_dx_matches_reference_on_cuda() -> None:
    _requires_cuda_extension()

    import torch

    from novadit.ops import adaln_ref, fused_adaln

    shape = (2, 3, 768)
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    scale = torch.randn(
        shape[0], shape[2], device="cuda", dtype=torch.float16, requires_grad=True
    )
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_scale = scale.detach().clone().requires_grad_(True)
    ref_shift = shift.detach().clone().requires_grad_(True)

    fused_adaln(x, scale, shift).backward(dy)
    adaln_ref(ref_x, ref_scale, ref_shift).backward(dy)

    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(scale.grad, ref_scale.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(shift.grad, ref_shift.grad, rtol=2e-2, atol=2e-2)
