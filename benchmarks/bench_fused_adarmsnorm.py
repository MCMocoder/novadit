# ruff: noqa: E402

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

import torch

from novadit.ops import (
    HAS_NOVADIT_CUDA,
    adarmsnorm_compile,
    adarmsnorm_ref,
    fused_adarmsnorm,
    fused_adarmsnorm_inference,
    novadit_cuda,
)

Shape = tuple[int, int, int]
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

    times_ms_sorted = sorted(times_ms)
    p90_idx = min(len(times_ms_sorted) - 1, int(len(times_ms_sorted) * 0.90))
    return (
        statistics.median(times_ms),
        statistics.mean(times_ms),
        times_ms_sorted[p90_idx],
    )


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 4e-2, 4e-2
    if dtype == torch.bfloat16:
        return 6e-2, 6e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


def relative_error_floor(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 1e-3
    if dtype == torch.bfloat16:
        return 1e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> ErrorMetrics:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    abs_diff = diff.abs()
    denominator = expected_f.abs().clamp_min(relative_error_floor(dtype))
    expected_l2 = torch.linalg.vector_norm(expected_f)

    return {
        "max_abs_err": abs_diff.max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_rel_err": (abs_diff / denominator).max().item(),
        "rel_l2_err": (
            torch.linalg.vector_norm(diff)
            / expected_l2.clamp_min(relative_error_floor(dtype))
        ).item(),
    }


def make_result(
    *,
    phase: str,
    dtype: torch.dtype,
    shape: Shape,
    ref_times: tuple[float, float, float],
    fused_times: tuple[float, float, float],
    errors: ErrorMetrics,
    compile_times: tuple[float, float, float] | None = None,
    compile_errors: ErrorMetrics | None = None,
) -> BenchmarkResult:
    ref_median, ref_mean, ref_p90 = ref_times
    fused_median, fused_mean, fused_p90 = fused_times
    result: BenchmarkResult = {
        "phase": phase,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": shape,
        "ref_median_ms": ref_median,
        "ref_mean_ms": ref_mean,
        "ref_p90_ms": ref_p90,
        "fused_median_ms": fused_median,
        "fused_mean_ms": fused_mean,
        "fused_p90_ms": fused_p90,
        "speedup": ref_median / fused_median if fused_median > 0 else math.inf,
        "compile_median_ms": None,
        "compile_mean_ms": None,
        "compile_p90_ms": None,
        "compile_speedup": None,
        "compile_errors": None,
        **errors,
    }
    if compile_times is not None:
        compile_median, compile_mean, compile_p90 = compile_times
        result["compile_median_ms"] = compile_median
        result["compile_mean_ms"] = compile_mean
        result["compile_p90_ms"] = compile_p90
        result["compile_speedup"] = ref_median / compile_median if compile_median > 0 else math.inf
    if compile_errors is not None:
        result["compile_errors"] = compile_errors
    return result


# ---- Forward benchmark ----

@torch.no_grad()
def run_forward_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    shift = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)

    def run_ref() -> torch.Tensor:
        return adarmsnorm_ref(x, scale, shift, eps=1e-5)

    def run_fused() -> torch.Tensor:
        return fused_adarmsnorm(x, scale, shift, eps=1e-5)

    def run_compile() -> torch.Tensor:
        return adarmsnorm_compile(x, scale, shift, eps=1e-5)

    ref = run_ref()
    out = run_fused()
    out_compile = run_compile()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="forward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=error_metrics(out, ref, dtype=dtype),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        compile_errors=error_metrics(out_compile, ref, dtype=dtype),
    )


# ---- Inference benchmark ----

@torch.no_grad()
def run_inference_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    shift = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)

    def run_ref() -> torch.Tensor:
        return adarmsnorm_ref(x, scale, shift, eps=1e-5)

    def run_fused() -> torch.Tensor:
        return fused_adarmsnorm_inference(x, scale, shift, eps=1e-5)

    def run_compile() -> torch.Tensor:
        return adarmsnorm_compile(x, scale, shift, eps=1e-5)

    ref = run_ref()
    out = run_fused()
    out_compile = run_compile()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="inference",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=error_metrics(out, ref, dtype=dtype),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        compile_errors=error_metrics(out_compile, ref, dtype=dtype),
    )


# ---- Full chain benchmark (forward + backward) ----

def run_full_chain_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(
        batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)
    inputs = (x, scale, shift)

    ref_y = adarmsnorm_ref(x, scale, shift, eps=1e-5)
    fused_y = fused_adarmsnorm(x, scale, shift, eps=1e-5)
    compile_y = adarmsnorm_compile(x, scale, shift, eps=1e-5)

    ref_outputs = ref_y, *torch.autograd.grad(ref_y, inputs, dy, allow_unused=False)
    fused_outputs = fused_y, *torch.autograd.grad(fused_y, inputs, dy, allow_unused=False)
    compile_outputs = compile_y, *torch.autograd.grad(compile_y, inputs, dy, allow_unused=False)

    rtol, atol = tolerances(dtype)
    for fo, ro in zip(fused_outputs, ref_outputs, strict=True):
        torch.testing.assert_close(fo, ro, rtol=rtol, atol=atol)
    for co, ro in zip(compile_outputs, ref_outputs, strict=True):
        torch.testing.assert_close(co, ro, rtol=rtol, atol=atol)

    component_errors = {
        name: error_metrics(fo, ro, dtype=dtype)
        for name, fo, ro in zip(
            ("y", "dx", "dscale", "dshift"), fused_outputs, ref_outputs, strict=True
        )
    }
    compile_component_errors = {
        name: error_metrics(co, ro, dtype=dtype)
        for name, co, ro in zip(
            ("y", "dx", "dscale", "dshift"), compile_outputs, ref_outputs, strict=True
        )
    }
    full_errors = {
        metric: max(errs[metric] for errs in component_errors.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    compile_full_errors = {
        metric: max(errs[metric] for errs in compile_component_errors.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }

    def _run_full(mode: str) -> tuple[torch.Tensor, ...]:
        x_i = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        scale_i = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        shift_i = torch.randn_like(scale_i, requires_grad=True)
        dy_i = torch.randn_like(x_i)
        inputs_i = (x_i, scale_i, shift_i)

        if mode == "ref":
            y = adarmsnorm_ref(x_i, scale_i, shift_i, eps=1e-5)
        elif mode == "fused":
            y = fused_adarmsnorm(x_i, scale_i, shift_i, eps=1e-5)
        else:
            y = adarmsnorm_compile(x_i, scale_i, shift_i, eps=1e-5)

        grads = torch.autograd.grad(y, inputs_i, dy_i, allow_unused=False)
        return y, *grads

    def run_ref() -> tuple[torch.Tensor, ...]:
        return _run_full("ref")

    def run_fused() -> tuple[torch.Tensor, ...]:
        return _run_full("fused")

    def run_compile() -> tuple[torch.Tensor, ...]:
        return _run_full("compile")

    return make_result(
        phase="full_chain",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=full_errors,
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        compile_errors=compile_full_errors,
    )


# ---- Backward component benchmarks ----

def run_backward_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> list[BenchmarkResult]:
    batch_size, _, hidden_size = shape
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(
        batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)

    ref_y = adarmsnorm_ref(x, scale, shift, eps=1e-5)
    fused_y = fused_adarmsnorm(x, scale, shift, eps=1e-5)
    compile_y = adarmsnorm_compile(x, scale, shift, eps=1e-5)
    inputs = (x, scale, shift)

    def run_ref_bwd_verify() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(ref_y, inputs, dy, allow_unused=False)

    def run_fused_bwd_verify() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(fused_y, inputs, dy, allow_unused=False)

    def run_compile_bwd_verify() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(compile_y, inputs, dy, allow_unused=False)

    ref_grads = run_ref_bwd_verify()
    fused_grads = run_fused_bwd_verify()
    compile_grads = run_compile_bwd_verify()
    rtol, atol = tolerances(dtype)
    for fg, rg in zip(fused_grads, ref_grads, strict=True):
        torch.testing.assert_close(fg, rg, rtol=rtol, atol=atol)
    for cg, rg in zip(compile_grads, ref_grads, strict=True):
        torch.testing.assert_close(cg, rg, rtol=rtol, atol=atol)

    component_errors = {
        name: error_metrics(fg, rg, dtype=dtype)
        for name, fg, rg in zip(("dx", "dscale", "dshift"), fused_grads, ref_grads, strict=True)
    }
    compile_component_errors = {
        name: error_metrics(cg, rg, dtype=dtype)
        for name, cg, rg in zip(("dx", "dscale", "dshift"), compile_grads, ref_grads, strict=True)
    }
    full_errors = {
        metric: max(errs[metric] for errs in component_errors.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    compile_full_errors = {
        metric: max(errs[metric] for errs in compile_component_errors.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }

    def _run_bwd(mode: str) -> tuple[torch.Tensor, ...]:
        x_i = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        scale_i = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        shift_i = torch.randn_like(scale_i, requires_grad=True)
        dy_i = torch.randn_like(x_i)
        inputs_i = (x_i, scale_i, shift_i)

        if mode == "ref":
            y = adarmsnorm_ref(x_i, scale_i, shift_i, eps=1e-5)
        elif mode == "fused":
            y = fused_adarmsnorm(x_i, scale_i, shift_i, eps=1e-5)
        else:
            y = adarmsnorm_compile(x_i, scale_i, shift_i, eps=1e-5)

        return torch.autograd.grad(y, inputs_i, dy_i, allow_unused=False)

    def run_ref_bwd() -> tuple[torch.Tensor, ...]:
        return _run_bwd("ref")

    def run_fused_bwd() -> tuple[torch.Tensor, ...]:
        return _run_bwd("fused")

    def run_compile_bwd() -> tuple[torch.Tensor, ...]:
        return _run_bwd("compile")

    full_result = make_result(
        phase="backward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref_bwd, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused_bwd, warmup=warmup, iters=iters),
        errors=full_errors,
        compile_times=cuda_event_benchmark(run_compile_bwd, warmup=warmup, iters=iters),
        compile_errors=compile_full_errors,
    )

    with torch.no_grad():
        _, saved_scale, inv_rms = novadit_cuda.fused_adarmsnorm_forward(
            x, scale, shift, 1e-5
        )

        def run_ref_dx() -> torch.Tensor:
            xhat = x.float() * inv_rms.unsqueeze(-1)
            dxhat = dy.float() * (1.0 + saved_scale.float().unsqueeze(1))
            mean_dxhat_xhat = (dxhat * xhat).mean(dim=-1, keepdim=True)
            return inv_rms.unsqueeze(-1) * (dxhat - xhat * mean_dxhat_xhat)

        def run_fused_dx() -> torch.Tensor:
            return novadit_cuda.fused_adarmsnorm_backward_dx(
                dy, x, saved_scale, inv_rms
            )

        ref_dx = run_ref_dx()
        fused_dx = run_fused_dx()
        torch.testing.assert_close(fused_dx, ref_dx.to(dtype), rtol=rtol, atol=atol)
        dx_result = make_result(
            phase="backward_dx",
            dtype=dtype,
            shape=shape,
            ref_times=cuda_event_benchmark(run_ref_dx, warmup=warmup, iters=iters),
            fused_times=cuda_event_benchmark(run_fused_dx, warmup=warmup, iters=iters),
            errors=error_metrics(fused_dx, ref_dx, dtype=dtype),
        )

        def run_ref_dparam() -> tuple[torch.Tensor, torch.Tensor]:
            xhat = x.float() * inv_rms.unsqueeze(-1)
            dscale = (dy.float() * xhat).sum(dim=1)
            dshift = dy.float().sum(dim=1)
            return dscale, dshift

        def run_fused_dparam() -> tuple[torch.Tensor, torch.Tensor]:
            return novadit_cuda.fused_adarmsnorm_backward_dparam(dy, x, inv_rms)

        ref_dscale, ref_dshift = run_ref_dparam()
        fused_dscale, fused_dshift = run_fused_dparam()
        torch.testing.assert_close(fused_dscale, ref_dscale, rtol=rtol, atol=atol)
        torch.testing.assert_close(fused_dshift, ref_dshift, rtol=rtol, atol=atol)
        dparam_result = make_result(
            phase="backward_dparam",
            dtype=dtype,
            shape=shape,
            ref_times=cuda_event_benchmark(run_ref_dparam, warmup=warmup, iters=iters),
            fused_times=cuda_event_benchmark(run_fused_dparam, warmup=warmup, iters=iters),
            errors=error_metrics(fused_dscale, ref_dscale, dtype=dtype),
        )

    return [full_result, dx_result, dparam_result]


# ---- Report formatting ----

def append_results_table(
    lines: list[str],
    phase: str,
    results: list[BenchmarkResult],
) -> None:
    phase_results = [r for r in results if r["phase"] == phase]
    if not phase_results:
        return

    lines.append(phase.replace("_", " ").title())
    lines.append("-" * 90)
    lines.append(
        f"{'dtype':<10} {'shape':<18} "
        f"{'ref_ms':>9} {'fused_ms':>9} {'compile_ms':>10} {'f_speedup':>9} {'c_speedup':>9}"
    )
    lines.append("-" * 90)

    for item in phase_results:
        compile_ms = ""
        compile_speedup = ""
        if item.get("compile_median_ms") is not None:
            compile_ms = f"{item['compile_median_ms']:>10.4f}"
            compile_speedup = f"{item['compile_speedup']:>8.2f}x"
        else:
            compile_ms = " " * 10
            compile_speedup = " " * 9
        lines.append(
            f"{item['dtype']:<10} {str(item['shape']):<18} "
            f"{item['ref_median_ms']:>9.4f} "
            f"{item['fused_median_ms']:>9.4f} "
            f"{compile_ms} "
            f"{item['speedup']:>8.2f}x "
            f"{compile_speedup}"
        )

    speedups = [float(r["speedup"]) for r in phase_results]
    lines.append(f"Summary: fused avg={statistics.mean(speedups):.2f}x, min={min(speedups):.2f}x, max={max(speedups):.2f}x")
    compile_results = [r for r in phase_results if r.get("compile_speedup") is not None]
    if compile_results:
        compile_speedups = [float(r["compile_speedup"]) for r in compile_results]
        lines.append(f"        compile avg={statistics.mean(compile_speedups):.2f}x, min={min(compile_speedups):.2f}x, max={max(compile_speedups):.2f}x")
    lines.append("")

    lines.append(f"{phase.replace('_', ' ').title()} Error Metrics")
    lines.append("-" * 120)
    lines.append(
        f"{'dtype':<10} {'shape':<18} {'type':<7} "
        f"{'max_abs':>13} {'rmse':>13} {'max_rel':>13} {'rel_l2':>13}"
    )
    lines.append("-" * 120)
    for item in phase_results:
        lines.append(
            f"{item['dtype']:<10} {str(item['shape']):<18} {'fused':<7} "
            f"{item['max_abs_err']:>13.4e} "
            f"{item['rmse']:>13.4e} "
            f"{item['max_rel_err']:>13.4e} "
            f"{item['rel_l2_err']:>13.4e}"
        )
        if item.get("compile_errors") is not None:
            ce = item["compile_errors"]
            lines.append(
                f"{item['dtype']:<10} {str(item['shape']):<18} {'compile':<7} "
                f"{ce['max_abs_err']:>13.4e} "
                f"{ce['rmse']:>13.4e} "
                f"{ce['max_rel_err']:>13.4e} "
                f"{ce['rel_l2_err']:>13.4e}"
            )
    lines.append("")


def format_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "NovaDiT fused AdaRMSNorm Benchmark: ref vs CUDA vs torch.compile",
        "=" * 72,
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Environment",
        "-" * 48,
        f"PyTorch version  : {torch.__version__}",
        f"CUDA version     : {torch.version.cuda}",
        f"GPU              : {torch.cuda.get_device_name(0)}",
        f"Device capability: {torch.cuda.get_device_capability(0)}",
        "",
    ]

    for phase in ("inference", "forward", "full_chain", "backward", "backward_dx", "backward_dparam"):
        append_results_table(lines, phase, results)

    lines.extend([
        "Notes",
        "-" * 48,
        "AdaRMSNorm = RMSNorm + adaptive scale/shift modulation (no centering).",
        "Inference returns y only without inv_rms buffer.",
        "Backward dx omits the mean_dxhat term (RMSNorm does not subtract mean).",
        "f_speedup: ref vs hand-written CUDA kernel; c_speedup: ref vs torch.compile.",
        "torch.compile uses dynamic=True with Triton auto-fusion via inductor backend.",
        "Latency values are CUDA event medians; allocations are included.",
    ])
    return "\n".join(lines)


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NovaDiT fused AdaRMSNorm CUDA kernels."
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["inference", "forward", "full_chain", "backward", "all"],
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("adarmsnorm_benchmark_report.txt"),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["fp16", "bf16"],
        choices=["fp16", "bf16"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not HAS_NOVADIT_CUDA:
        raise RuntimeError("novadit_cuda extension is not built/importable.")

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

    shapes: list[Shape] = [
        (8, 256, 768),
        (4, 1024, 1152),
        (8, 256, 770),
    ]
    results: list[BenchmarkResult] = []

    for dtype_name in args.dtypes:
        dtype = parse_dtype(dtype_name)
        for shape in shapes:
            if args.mode in ("inference", "all"):
                print(f"Benchmarking inference dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_inference_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )
            if args.mode in ("forward", "all"):
                print(f"Benchmarking forward dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_forward_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )
            if args.mode in ("full_chain", "all"):
                print(f"Benchmarking full chain dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_full_chain_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )
            if args.mode in ("backward", "all"):
                print(f"Benchmarking backward dtype={dtype_name}, shape={shape} ...")
                results.extend(
                    run_backward_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )

    report = format_report(results)
    args.out.write_text(report, encoding="utf-8")
    print("")
    print(report)
    print("")
    print(f"TXT report written to: {args.out}")


if __name__ == "__main__":
    main()
