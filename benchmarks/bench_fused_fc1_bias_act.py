from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "csrc" / "cute_dsl" / "fc1_bias_act"))

import torch
import torch.nn.functional as F

from novadit.ops import (
    fc1_bias_act_ref,
    interleave_weight,
    split_gated_weight,
)

BenchmarkResult = dict[str, Any]
ErrorMetrics = dict[str, float]


def cuda_event_benchmark(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    sorted_times = sorted(times_ms)
    p90_idx = min(len(sorted_times) - 1, int(len(sorted_times) * 0.90))
    return statistics.median(sorted_times), statistics.mean(sorted_times), sorted_times[p90_idx]


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    abs_diff = diff.abs()
    denominator = expected_f.abs().clamp_min(1e-6)
    expected_l2 = torch.linalg.vector_norm(expected_f)
    return {
        "max_abs_err": abs_diff.max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_rel_err": (abs_diff / denominator).max().item(),
        "rel_l2_err": (torch.linalg.vector_norm(diff) / expected_l2.clamp_min(1e-6)).item(),
    }


def _make_cute_3d(t: torch.Tensor, cute_dtype=None) -> tuple:
    if cute_dtype is None:
        import cutlass
        cute_dtype = cutlass.Float16
    import cutlass.torch as ct
    cute_t, _ = ct.cute_tensor_like(
        t.unsqueeze(-1).cpu(), cute_dtype, False, 16,
    )
    return cute_t


def _build_cute_kernel(
    M: int, H: int, N: int, act_id: int, dtype: object,
) -> tuple:
    """Compile the CuTe DSL kernel once. Returns (compiled, make_args_fn)."""
    import cuda.bindings.driver  # noqa: F401
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from fc1_bias_act import FC1GatedActSM120BF16Kernel, FusedGatedMlpShape

    torch_dtype = torch.float16 if dtype == cutlass.Float16 else torch.bfloat16

    x_cute = _make_cute_3d(torch.empty(M, H, dtype=torch_dtype), dtype)
    w_gate_cute = _make_cute_3d(torch.empty(N, H, dtype=torch_dtype), dtype)
    w_value_cute = _make_cute_3d(torch.empty(N, H, dtype=torch_dtype), dtype)
    y_cute = _make_cute_3d(torch.empty(M, N, dtype=torch_dtype), dtype)

    kernel = FC1GatedActSM120BF16Kernel(
        FusedGatedMlpShape(batch_tokens=M, hidden_size=H, intermediate_size=N),
        ab_dtype=dtype,
        output_dtype=dtype,
    )
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    stream = ct.default_stream()

    compiled = cute.compile(
        kernel, x_cute, w_gate_cute, w_value_cute, y_cute,
        act_id, max_clusters, stream,
    )

    def make_args(
        x: torch.Tensor, wg: torch.Tensor, wv: torch.Tensor, y: torch.Tensor,
    ):
        import cutlass
        import cutlass.torch as ct

        def _wrap(t: torch.Tensor):
            t3d = t.unsqueeze(-1).contiguous()
            return ct.from_dlpack(t3d, assumed_align=16), t3d

        x_c, _ = _wrap(x)
        wg_c, _ = _wrap(wg)
        wv_c, _ = _wrap(wv)
        y_c, y3d = _wrap(y)
        return x_c, wg_c, wv_c, y_c, y3d, stream

    return compiled, make_args


@torch.no_grad()
def run_case(
    *,
    B: int, S: int, H: int, I: int,
    activation: str,
    dtype: torch.dtype,
    cute_dtype: object,
    warmup: int, iters: int,
    run_cute: bool,
    cute_compiled: object | None = None,
    cute_args_fn: object | None = None,
) -> BenchmarkResult:
    M = B * S
    act_name = activation

    # Scale inputs to keep values within FP16/BF16 range during K-dim accumulation.
    scale = 1.0 / (H ** 0.5)
    x = torch.randn(M, H, device="cuda", dtype=dtype) * scale
    weight = torch.randn(2 * I, H, device="cuda", dtype=dtype) * scale

    # ── cuBLASLt baseline ─────────────────────────────────────────────
    def run_cublaslt() -> torch.Tensor:
        y_linear = F.linear(x, weight)
        gate, value = y_linear.chunk(2, dim=-1)
        if act_name == "silu":
            return value * F.silu(gate)
        else:
            return value * F.gelu(gate, approximate="tanh")

    ref = run_cublaslt()

    # ── CuTe DSL JIT ──────────────────────────────────────────────────
    cute_ok = False
    cute_median = cute_mean = cute_p90 = -1.0
    errors_cute: dict[str, float] = {}

    if run_cute and cute_compiled is not None and cute_args_fn is not None:
        w_inter = interleave_weight(weight)
        w_gate, w_value = split_gated_weight(w_inter)

        def run_cute_fn():
            x_c, wg_c, wv_c, y_c, y3d, stream = cute_args_fn(x, w_gate, w_value,
                torch.empty(M, I, device="cuda", dtype=dtype))
            cute_compiled(x_c, wg_c, wv_c, y_c, stream)
            return y3d.squeeze(-1)

        out_cute = run_cute_fn()
        torch.cuda.synchronize()

        rtol = 3e-2
        try:
            torch.testing.assert_close(out_cute, ref, atol=rtol, rtol=rtol)
            cute_ok = True
        except AssertionError:
            print(f"  WARNING: CuTe DSL exceeds tolerance (rtol={rtol})")

        if cute_ok:
            errors_cute = error_metrics(out_cute, ref)
            cute_median, cute_mean, cute_p90 = cuda_event_benchmark(
                run_cute_fn, warmup=warmup, iters=iters
            )

    ref_median, ref_mean, ref_p90 = cuda_event_benchmark(
        run_cublaslt, warmup=warmup, iters=iters
    )

    result: BenchmarkResult = {
        "dtype": str(dtype).replace("torch.", ""),
        "shape": (B, S, H),
        "I": I, "M": M,
        "activation": activation,
        "cublaslt_ms": ref_median,
        "cublaslt_mean": ref_mean,
        "cublaslt_p90": ref_p90,
        "cute_ms": cute_median,
        "cute_speedup": ref_median / cute_median if cute_median > 0 else 0,
        **errors_cute,
    }
    return result


def _build_cute_backward_dx_kernel(
    M: int, H: int, I: int, dtype: object,
) -> tuple:
    """Compile the backward dx CuTe DSL kernel once. Returns (compiled, make_args_fn).
    
    Weights are passed transposed ([H, I] instead of [I, H]) so the K dimension
    aligns with the A operands (dgate/dvalue have K=I).
    """
    import cuda.bindings.driver  # noqa: F401
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from fc1_bias_act import FC1GatedActSM120BF16BackwardDxKernel

    torch_dtype = torch.float16 if dtype == cutlass.Float16 else torch.bfloat16

    dgate_cute = _make_cute_3d(torch.empty(M, I, dtype=torch_dtype), dtype)
    dvalue_cute = _make_cute_3d(torch.empty(M, I, dtype=torch_dtype), dtype)
    w_gate_T_cute = _make_cute_3d(torch.empty(H, I, dtype=torch_dtype), dtype)
    w_value_T_cute = _make_cute_3d(torch.empty(H, I, dtype=torch_dtype), dtype)
    dx_cute = _make_cute_3d(torch.empty(M, H, dtype=torch_dtype), dtype)

    kernel = FC1GatedActSM120BF16BackwardDxKernel(
        ab_dtype=dtype,
        output_dtype=dtype,
    )
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    stream = ct.default_stream()

    compiled = cute.compile(
        kernel,
        dgate_cute, dvalue_cute, w_gate_T_cute, w_value_T_cute, dx_cute,
        max_clusters, stream,
    )

    def make_args(
        dg: torch.Tensor, dv: torch.Tensor,
        wg_T: torch.Tensor, wv_T: torch.Tensor,
        dx: torch.Tensor,
    ):
        import cutlass.torch as ct

        def _wrap(t: torch.Tensor):
            t3d = t.unsqueeze(-1).contiguous()
            return ct.from_dlpack(t3d, assumed_align=16), t3d

        dg_c, _ = _wrap(dg)
        dv_c, _ = _wrap(dv)
        wg_c, _ = _wrap(wg_T)
        wv_c, _ = _wrap(wv_T)
        dx_c, dx3d = _wrap(dx)
        return dg_c, dv_c, wg_c, wv_c, dx_c, dx3d, stream

    return compiled, make_args


@torch.no_grad()
def run_backward_dx_case(
    *,
    B: int, S: int, H: int, I: int,
    dtype: torch.dtype,
    cute_dtype: object,
    warmup: int, iters: int,
    run_cute: bool,
    cute_compiled: object | None = None,
    cute_args_fn: object | None = None,
) -> BenchmarkResult:
    M = B * S

    scale = 1.0 / (H ** 0.5)
    dgate = torch.randn(M, I, device="cuda", dtype=dtype) * scale
    dvalue = torch.randn(M, I, device="cuda", dtype=dtype) * scale
    weight = torch.randn(2 * I, H, device="cuda", dtype=dtype) * scale

    w_inter = interleave_weight(weight)
    w_gate, w_value = split_gated_weight(w_inter)

    # ── baseline: two torch.mm + add ───────────────────────────────────
    def run_baseline() -> torch.Tensor:
        return torch.mm(dgate.float(), w_gate.float()) + \
               torch.mm(dvalue.float(), w_value.float())

    ref = run_baseline().to(dtype)

    # ── CuTe DSL JIT ──────────────────────────────────────────────────
    cute_ok = False
    cute_median = cute_mean = cute_p90 = -1.0
    errors_cute: dict[str, float] = {}

    if run_cute and cute_compiled is not None and cute_args_fn is not None:
        w_gate_T = w_gate.T.contiguous()
        w_value_T = w_value.T.contiguous()

        def run_cute_fn():
            dg_c, dv_c, wg_c, wv_c, dx_c, dx3d, stream = cute_args_fn(
                dgate, dvalue, w_gate_T, w_value_T,
                torch.empty(M, H, device="cuda", dtype=dtype))
            cute_compiled(dg_c, dv_c, wg_c, wv_c, dx_c, stream)
            return dx3d.squeeze(-1)

        out_cute = run_cute_fn()
        torch.cuda.synchronize()

        rtol = 3e-2 if dtype == torch.bfloat16 else 2e-2
        try:
            torch.testing.assert_close(out_cute, ref, atol=rtol, rtol=rtol)
            cute_ok = True
        except AssertionError:
            print(f"  WARNING: CuTe DSL backward dx exceeds tolerance (rtol={rtol})")

        if cute_ok:
            errors_cute = error_metrics(out_cute, ref)
            cute_median, cute_mean, cute_p90 = cuda_event_benchmark(
                run_cute_fn, warmup=warmup, iters=iters
            )

    ref_median, ref_mean, ref_p90 = cuda_event_benchmark(
        run_baseline, warmup=warmup, iters=iters
    )

    result: BenchmarkResult = {
        "dtype": str(dtype).replace("torch.", ""),
        "shape": (B, S, H),
        "I": I, "M": M,
        "pass_name": "backward_dx",
        "baseline_ms": ref_median,
        "baseline_mean": ref_mean,
        "baseline_p90": ref_p90,
        "cute_ms": cute_median,
        "cute_speedup": ref_median / cute_median if cute_median > 0 else 0,
        **errors_cute,
    }
    return result


def _format_result_section(
    lines: list[str],
    results: list[BenchmarkResult],
    title: str,
    baseline_label: str,
) -> None:
    if not results:
        return
    lines.append("")
    lines.append(title)
    lines.append("-" * 80)
    header = f"{'shape(B,S,H,I)':<22} {'dt':<6} {'baseline':>10} {'CuTeDSL':>10} {'speedup':>9}"
    if results[0].get("activation"):
        header = f"{'shape(B,S,H,I)':<22} {'dt':<6} {'act':<5} {baseline_label:>10} {'CuTeDSL':>10} {'speedup':>9}"
    lines.append(header)
    lines.append("-" * 80)
    for r in results:
        shape_str = f"({r['shape'][0]},{r['shape'][1]},{r['shape'][2]},{r['I']})"
        dt = r.get("dtype", "float16")
        act = r.get("activation", "")
        bl_ms = r.get("baseline_ms", r.get("cublaslt_ms", 0))
        if r.get("cute_ms", -1) > 0:
            if act:
                lines.append(
                    f"{shape_str:<22} {dt:<6} {act:<5} "
                    f"{bl_ms:>10.3f} {r['cute_ms']:>10.3f} "
                    f"{r['cute_speedup']:>8.2f}x"
                )
            else:
                lines.append(
                    f"{shape_str:<22} {dt:<6} "
                    f"{bl_ms:>10.3f} {r['cute_ms']:>10.3f} "
                    f"{r['cute_speedup']:>8.2f}x"
                )
        else:
            if act:
                lines.append(
                    f"{shape_str:<22} {dt:<6} {act:<5} "
                    f"{bl_ms:>10.3f} {'N/A':>10} {'-':>9}"
                )
            else:
                lines.append(
                    f"{shape_str:<22} {dt:<6} "
                    f"{bl_ms:>10.3f} {'N/A':>10} {'-':>9}"
                )

    speedups = [float(r["cute_speedup"]) for r in results if r.get("cute_speedup", 0) > 0]
    if speedups:
        lines.append(
            f"\n  avg={statistics.mean(speedups):.2f}x "
            f"min={min(speedups):.2f}x max={max(speedups):.2f}x"
        )

    err_items = [r for r in results if r.get("max_abs_err") is not None and r["max_abs_err"] > 0]
    if err_items:
        lines.extend([
            "", f"  Error ({title})",
            f"  {'shape':<22} {'max_abs':>10} {'rmse':>10} {'max_rel':>10}",
        ])
        for r in err_items:
            shape_str = f"({r['shape'][0]},{r['shape'][1]},{r['shape'][2]},{r['I']})"
            lines.append(
                f"  {shape_str:<22} "
                f"{r['max_abs_err']:>10.4e} {r['rmse']:>10.4e} "
                f"{r['max_rel_err']:>10.4e}"
            )


def format_report(
    fwd_results: list[BenchmarkResult],
    bw_results: list[BenchmarkResult] | None = None,
) -> str:
    lines = [
        "FC1 + Gated-Act : cuBLASLt vs CuTe DSL (SM120, m16n8k16)",
        "=" * 65,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Environment",
        "-" * 65,
        f"PyTorch  : {torch.__version__}",
        f"CUDA     : {torch.version.cuda}",
        f"GPU      : {torch.cuda.get_device_name(0)}",
        f"Capability: {torch.cuda.get_device_capability(0)}",
        "",
        "Latency (ms, median)",
        "-" * 80,
    ]

    _format_result_section(lines, fwd_results, "Forward: FC1 + Gated-Act", "cuBLASLt")

    if bw_results:
        _format_result_section(lines, bw_results, "Backward dx: dgate@w_gate + dvalue@w_value", "torch.mm")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cuBLASLt vs CuTe DSL FC1+gated-act (SM120, m16n8k16)."
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-cute", action="store_true",
                        help="Skip CuTe DSL, report cuBLASLt only")
    parser.add_argument("--dtypes", nargs="+", default=["fp16"],
                        choices=["fp16", "bf16"],
                        help="Data types to benchmark (default: fp16)")
    parser.add_argument("--no-backward", action="store_true",
                        help="Skip backward dx kernel benchmark")
    parser.add_argument("--out", type=Path,
                        default=Path("fc1_bias_act_cute_bench.txt"))
    args = parser.parse_args()
    run_cute = not args.no_cute
    run_backward_dx = run_cute and not args.no_backward

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    import cutlass

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

    shapes = [
        (1, 4096, 768, 3072),
        (2, 1024, 768, 768),
        (2, 1024, 1152, 3072),
        (4, 4096, 1536, 4096),
        (1, 8192, 768, 3072),
        (8, 512, 1536, 4096),
    ]

    # Pre-compile CuTe DSL kernel for each unique (dtype,M,H,I,act) tuple.
    cute_kernels: dict[tuple, tuple] = {}
    if run_cute:
        seen: set[tuple] = set()
        for dt_name in args.dtypes:
            torch_dt = torch.float16 if dt_name == "fp16" else torch.bfloat16
            cute_dt = cutlass.Float16 if dt_name == "fp16" else cutlass.BFloat16
            for B, S, H, I in shapes:
                for act in ("silu", "gelu"):
                    key = (dt_name, B * S, H, I, act)
                    if key in seen:
                        continue
                    seen.add(key)
                    act_id = 0 if act == "silu" else 1
                    print(f"Compiling CuTe DSL kernel: {dt_name}, M={key[1]}, H={key[2]}, I={key[3]}, {act} ...")
                    cute_kernels[key] = _build_cute_kernel(key[1], key[2], key[3], act_id, cute_dt)

    results: list[BenchmarkResult] = []
    for dt_name in args.dtypes:
        torch_dt = torch.float16 if dt_name == "fp16" else torch.bfloat16
        cute_dt = cutlass.Float16 if dt_name == "fp16" else cutlass.BFloat16
        for B, S, H, I in shapes:
            for act in ("silu", "gelu"):
                key = (dt_name, B * S, H, I, act)
                comp, args_fn = cute_kernels.get(key, (None, None))
                print(f"[{dt_name}] B={B},S={S},H={H},I={I} act={act} ...")
                results.append(
                    run_case(
                        B=B, S=S, H=H, I=I,
                        activation=act,
                        dtype=torch_dt,
                        cute_dtype=cute_dt,
                        warmup=args.warmup, iters=args.iters,
                        run_cute=run_cute,
                        cute_compiled=comp,
                        cute_args_fn=args_fn,
                    )
                )
                torch.cuda.empty_cache()

    # ── Backward dx ───────────────────────────────────────────────────
    bw_kernels: dict[tuple, tuple] = {}
    if run_backward_dx:
        bw_seen: set[tuple] = set()
        for dt_name in args.dtypes:
            cute_dt = cutlass.Float16 if dt_name == "fp16" else cutlass.BFloat16
            for B, S, H, I in shapes:
                key = (dt_name, B * S, H, I)
                if key in bw_seen:
                    continue
                bw_seen.add(key)
                print(f"Compiling backward dx CuTe kernel: {dt_name}, M={key[1]}, H={key[2]}, I={key[3]} ...")
                bw_kernels[key] = _build_cute_backward_dx_kernel(key[1], key[2], key[3], cute_dt)

    bw_results: list[BenchmarkResult] = []
    if run_backward_dx:
        for dt_name in args.dtypes:
            torch_dt = torch.float16 if dt_name == "fp16" else torch.bfloat16
            cute_dt = cutlass.Float16 if dt_name == "fp16" else cutlass.BFloat16
            for B, S, H, I in shapes:
                key = (dt_name, B * S, H, I)
                comp, args_fn = bw_kernels.get(key, (None, None))
                print(f"[{dt_name}] backward_dx B={B},S={S},H={H},I={I} ...")
                bw_results.append(
                    run_backward_dx_case(
                        B=B, S=S, H=H, I=I,
                        dtype=torch_dt,
                        cute_dtype=cute_dt,
                        warmup=args.warmup, iters=args.iters,
                        run_cute=run_backward_dx,
                        cute_compiled=comp,
                        cute_args_fn=args_fn,
                    )
                )
                torch.cuda.empty_cache()

    report = format_report(results, bw_results)
    args.out.write_text(report, encoding="utf-8")
    print("")
    print(report)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
