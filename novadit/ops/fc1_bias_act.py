from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import novadit_cuda

_FW_CACHE: dict[tuple, Any] = {}
_BW_DX_CACHE: dict[tuple, Any] = {}

_CUTE_TILE_M = 64
_CUTE_TILE_N = 64


def _optimal_clusters(M: int, N: int, sm_count: int) -> int:
    """Pick max_active_clusters for the persistent tile scheduler.

    Small shapes (≤128 work tiles): run multiple CTAs in parallel so SMs
    are not idle.  Large shapes: keep 1 CTA so the persistent grid avoids
    launch overhead on thousands of tiles.
    """
    total_tiles = (M // _CUTE_TILE_M) * (N // _CUTE_TILE_N)
    if total_tiles <= 128:
        return min(total_tiles, sm_count)
    return 1


def interleave_weight(weight: torch.Tensor) -> torch.Tensor:
    I = weight.shape[0] // 2
    w_gate = weight[:I]
    w_value = weight[I:]
    interleaved = torch.stack([w_gate, w_value], dim=1)
    return interleaved.reshape(2 * I, -1).contiguous()


def un_interleave_weight(weight: torch.Tensor) -> torch.Tensor:
    I = weight.shape[0] // 2
    w_gate = weight[0::2]
    w_value = weight[1::2]
    return torch.cat([w_gate, w_value], dim=0).contiguous()


def interleave_bias(bias: torch.Tensor) -> torch.Tensor:
    I = bias.shape[0] // 2
    return torch.stack([bias[:I], bias[I:]], dim=1).reshape(2 * I).contiguous()


def un_interleave_bias(bias: torch.Tensor) -> torch.Tensor:
    I = bias.shape[0] // 2
    return torch.cat([bias[0::2], bias[1::2]], dim=0).contiguous()


def fc1_bias_act_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    z = F.linear(x, weight, bias)
    if activation == "silu":
        z_gate, z_value = z.chunk(2, dim=-1)
        return z_value * F.silu(z_gate)
    elif activation == "gelu":
        z_gate, z_value = z.chunk(2, dim=-1)
        return z_value * F.gelu(z_gate, approximate="tanh")
    raise ValueError(f"Unsupported activation: {activation}")


def _get_forward_kernel(
    M: int, H: int, I: int, act_id: int, dtype: torch.dtype,
) -> Any | None:
    """Lazily compile and cache the CuTe DSL forward kernel.

    Returns ``(compiled, make_args_fn)`` or ``None`` if CuTe DSL is unavailable.
    """
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as ct
        from csrc.cute_dsl.fc1_bias_act import FC1GatedActSM120BF16Kernel, FusedGatedMlpShape
    except ImportError:
        return None

    cute_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    hw = cutlass.utils.HardwareInfo()
    sm_count = hw.get_device_multiprocessor_count()
    clusters = _optimal_clusters(M, I, sm_count)  # N = I for forward (output dim)

    cache_key = (M, H, I, act_id, clusters, dtype)
    if cache_key in _FW_CACHE:
        return _FW_CACHE[cache_key]

    try:
        torch_dtype = dtype
        def _empty(m, n):
            t = torch.empty(m, n, dtype=torch_dtype)
            cute_t, _ = ct.cute_tensor_like(t.unsqueeze(-1).cpu(), cute_dtype, False, 16)
            return cute_t

        kernel = FC1GatedActSM120BF16Kernel(
            FusedGatedMlpShape(batch_tokens=M, hidden_size=H, intermediate_size=I),
            ab_dtype=cute_dtype, output_dtype=cute_dtype,
        )
        compile_stream = ct.default_stream()

        compiled = cute.compile(
            kernel,
            _empty(M, H), _empty(I, H), _empty(I, H), _empty(M, I),
            act_id, clusters, compile_stream,
        )

        def make_args(x, wg, wv, y):
            def _wrap(t):
                t3d = t.unsqueeze(-1).contiguous()
                return ct.from_dlpack(t3d, assumed_align=16), t3d
            xc, _ = _wrap(x)
            wgc, _ = _wrap(wg)
            wvc, _ = _wrap(wv)
            yc, y3d = _wrap(y)
            return xc, wgc, wvc, yc, y3d

        _FW_CACHE[cache_key] = (compiled, make_args)
        return _FW_CACHE[cache_key]
    except Exception:
        _FW_CACHE[cache_key] = None
        return None


def _get_backward_dx_kernel(
    M: int, H: int, I: int, dtype: torch.dtype,
) -> Any | None:
    """Lazily compile and cache the CuTe DSL backward dx kernel.

    Returns ``(compiled, make_args_fn)`` or ``None`` if CuTe DSL is unavailable.
    """
    try:
        import cuda.bindings.driver  # noqa: F401
    except ImportError:
        return None

    try:
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as ct
        from csrc.cute_dsl.fc1_bias_act import FC1GatedActSM120BF16BackwardDxKernel
    except ImportError:
        return None

    cute_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    hw = cutlass.utils.HardwareInfo()
    sm_count = hw.get_device_multiprocessor_count()
    clusters = _optimal_clusters(M, H, sm_count)  # N = H for backward dx (output dim)

    cache_key = (M, H, I, clusters, dtype)
    if cache_key in _BW_DX_CACHE:
        return _BW_DX_CACHE[cache_key]

    try:
        torch_dtype = dtype

        def _make_cute_3d(m: int, n: int) -> Any:
            t = torch.empty(m, n, dtype=torch_dtype)
            cute_t, _ = ct.cute_tensor_like(
                t.unsqueeze(-1).cpu(), cute_dtype, False, 16,
            )
            return cute_t

        dgate_cute = _make_cute_3d(M, I)
        dvalue_cute = _make_cute_3d(M, I)
        w_gate_T_cute = _make_cute_3d(H, I)
        w_value_T_cute = _make_cute_3d(H, I)
        dx_cute = _make_cute_3d(M, H)

        kernel = FC1GatedActSM120BF16BackwardDxKernel(
            ab_dtype=cute_dtype,
            output_dtype=cute_dtype,
        )
        compile_stream = ct.default_stream()

        compiled = cute.compile(
            kernel,
            dgate_cute, dvalue_cute, w_gate_T_cute, w_value_T_cute, dx_cute,
            clusters, compile_stream,
        )

        def make_args(
            dg: torch.Tensor, dv: torch.Tensor,
            wg_T: torch.Tensor, wv_T: torch.Tensor,
            dx: torch.Tensor,
        ):
            def _wrap(t: torch.Tensor):
                t3d = t.unsqueeze(-1).contiguous()
                return ct.from_dlpack(t3d, assumed_align=16), t3d

            dg_c, _ = _wrap(dg)
            dv_c, _ = _wrap(dv)
            wg_c, _ = _wrap(wg_T)
            wv_c, _ = _wrap(wv_T)
            dx_c, dx3d = _wrap(dx)
            return dg_c, dv_c, wg_c, wv_c, dx_c, dx3d

        _BW_DX_CACHE[cache_key] = (compiled, make_args)
        return _BW_DX_CACHE[cache_key]

    except Exception:
        _BW_DX_CACHE[cache_key] = None
        return None


class _FusedFc1BiasActFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        act_id: int,
    ) -> torch.Tensor:
        B, S, H = x.shape
        M = B * S
        I = weight.shape[0] // 2

        cached = _get_forward_kernel(M, H, I, act_id, x.dtype)
        if cached is not None and bias is None:
            compiled, make_args = cached
            w_gate = weight[:I].contiguous()
            w_value = weight[I:].contiguous()
            y = torch.empty(B, S, I, device=x.device, dtype=x.dtype)

            import cutlass.torch as ct
            xc, wgc, wvc, yc, y3d = make_args(x.reshape(M, H), w_gate, w_value, y.reshape(M, I))
            compiled(xc, wgc, wvc, yc, ct.current_stream())
        elif novadit_cuda is not None and x.dtype == torch.bfloat16:
            w_inter = interleave_weight(weight)
            b_inter = interleave_bias(bias) if bias is not None else None
            y = novadit_cuda.fused_fc1_bias_act_forward(x, w_inter, b_inter, act_id)
        else:
            y = fc1_bias_act_ref(x, weight, bias, "silu" if act_id == 0 else "gelu")

        ctx.save_for_backward(x, weight, bias)
        ctx.act_id = act_id
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        x, weight, bias = ctx.saved_tensors
        act_id = ctx.act_id

        z = F.linear(x, weight, bias)
        z.requires_grad_(True)
        z.retain_grad()

        if act_id == 0:
            z_gate, z_value = z.chunk(2, dim=-1)
            y_ref = F.silu(z_gate) * z_value
        else:
            z_gate, z_value = z.chunk(2, dim=-1)
            y_ref = F.gelu(z_gate, approximate="tanh") * z_value

        y_ref.backward(dy)
        dz = z.grad

        M = x.shape[0] * x.shape[1]
        dz_flat = dz.reshape(M, -1)
        x_flat = x.reshape(M, -1)

        I = weight.shape[0] // 2
        H = weight.shape[1]

        dx_flat = _backward_dx_cute(dz_flat, weight, M, H, I)
        dW = dz_flat.T @ x_flat

        if bias is not None:
            dbias = dz_flat.sum(dim=0)
        else:
            dbias = None

        dx = dx_flat.reshape(x.shape)
        return dx, dW, dbias, None


def _backward_dx_cute(
    dz_flat: torch.Tensor,
    weight: torch.Tensor,
    M: int,
    H: int,
    I: int,
) -> torch.Tensor:
    """Compute dx = dz @ weight using the CuTe DSL kernel if available.

    Falls back to a standard torch matmul when the kernel is unavailable or the
    shape/dtype combination is unsupported.
    """
    if not dz_flat.is_cuda or dz_flat.dtype not in (torch.float16, torch.bfloat16):
        return dz_flat @ weight

    cached = _get_backward_dx_kernel(M, H, I, dz_flat.dtype)
    if cached is None:
        return dz_flat @ weight

    compiled, make_args = cached

    dgate = dz_flat[:, :I].contiguous()
    dvalue = dz_flat[:, I:].contiguous()

    w_gate = weight[:I].contiguous()
    w_value = weight[I:].contiguous()
    w_gate_T = w_gate.T.contiguous()
    w_value_T = w_value.T.contiguous()

    dx = torch.empty(M, H, device=dz_flat.device, dtype=dz_flat.dtype)

    import cutlass.torch as ct
    dg_c, dv_c, wg_c, wv_c, dx_c, dx3d = make_args(
        dgate, dvalue, w_gate_T, w_value_T, dx,
    )
    stream = ct.current_stream()
    compiled(dg_c, dv_c, wg_c, wv_c, dx_c, stream)

    return dx3d.squeeze(-1)


def split_gated_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an interleaved gated FC1 weight into (w_gate, w_value).

    ``interleave_weight`` produces a ``[2*I, H]`` tensor whose even rows are the
    gate half and odd rows are the value half. This helper returns the two
    ``[I, H]`` contiguous halves used by the CuTe DSL kernel's two-weight ABI.
    The input is expected to already be interleaved; a plain ``[2*I, H]`` weight
    (gate stacked over value) is also accepted and treated as interleaved.
    """
    if weight.dim() != 2 or weight.shape[0] % 2 != 0:
        raise ValueError(
            f"weight must have shape [2*I, H] with an even first dim, got {tuple(weight.shape)}"
        )
    i = weight.shape[0] // 2
    w_gate = weight[0::2].contiguous()
    w_value = weight[1::2].contiguous()
    if w_gate.shape[0] != i or w_value.shape[0] != i:
        raise ValueError("gate/value split produced mismatched shapes")
    return w_gate, w_value


def fused_fc1_bias_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    if activation not in {"silu", "gelu"}:
        raise ValueError(f"Unsupported activation: {activation}")

    if (
        novadit_cuda is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and (bias is None or bias.dtype == x.dtype)
    ):
        act_id = 0 if activation == "silu" else 1
        return _FusedFc1BiasActFunction.apply(x, weight, bias, act_id)
    return fc1_bias_act_ref(x, weight, bias, activation)
