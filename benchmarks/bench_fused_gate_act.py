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
    fused_geglu,
    fused_gate_act,
    gate_act_ref,
    geglu_ref,
)

Shape = tuple[int, int, int]  # (B, S, 2*I)
BenchmarkResult = dict[str, Any]
ErrorMetrics = dict[str, float]


def cuda_event_benchmark(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    """Return median, mean, and p90 latency in milliseconds."""
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
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float16:
        return 2e-2, 2e-2
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> ErrorMetrics:
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
        "rel_l2_err": (
            torch.linalg.vector_norm(diff) / expected_l2.clamp_min(1e-6)
        ).item(),
    }


def make_result(
    *,
    phase: str,
    dtype: torch.dtype,
    shape: Shape,
    inter_size: int,
    ref_times: tuple[float, float, float],
    fused_times: tuple[float, float, float],
    errors: ErrorMetrics,
) -> BenchmarkResult:
    ref_median, ref_mean, ref_p90 = ref_times
    fused_median, fused_mean, fused_p90 = fused_times
    return {
        "phase": phase,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": shape,
        "inter_size": inter_size,
        "ref_median_ms": ref_median,
        "ref_mean_ms": ref_mean,
        "ref_p90_ms": ref_p90,
        "fused_median_ms": fused_median,
        "fused_mean_ms": fused_mean,
        "fused_p90_ms": fused_p90,
        "speedup": ref_median / fused_median if fused_median > 0 else math.inf,
        **errors,
    }


# ---- GeGLU benchmark ----

@torch.no_grad()
def run_geglu_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    z = torch.randn(shape, device="cuda", dtype=dtype)
    inter_size = shape[2] // 2

    def run_ref() -> torch.Tensor:
        return geglu_ref(z)

    def run_fused() -> torch.Tensor:
        return fused_geglu(z)

    ref = run_ref()
    out = run_fused()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="geglu",
        dtype=dtype,
        shape=shape,
        inter_size=inter_size,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=error_metrics(out, ref),
    )


# ---- SiLU-gated benchmark ----

@torch.no_grad()
def run_silu_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    inter_size = shape[2] // 2

    def run_ref() -> torch.Tensor:
        return gate_act_ref(x, activation="silu")

    def run_fused() -> torch.Tensor:
        return fused_gate_act(x, activation="silu")

    ref = run_ref()
    out = run_fused()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="silu_gate",
        dtype=dtype,
        shape=shape,
        inter_size=inter_size,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=error_metrics(out, ref),
    )


# ---- GELU-gated via gate_act benchmark ----

@torch.no_grad()
def run_gelu_gate_case(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    inter_size = shape[2] // 2

    def run_ref() -> torch.Tensor:
        return gate_act_ref(x, activation="gelu")

    def run_fused() -> torch.Tensor:
        return fused_gate_act(x, activation="gelu")

    ref = run_ref()
    out = run_fused()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="gelu_gate",
        dtype=dtype,
        shape=shape,
        inter_size=inter_size,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        errors=error_metrics(out, ref),
    )


# ---- Report formatting ----

def append_phase_table(
    lines: list[str],
    phase: str,
    label: str,
    results: list[BenchmarkResult],
) -> None:
    phase_results = [r for r in results if r["phase"] == phase]
    if not phase_results:
        return

    lines.append(label)
    lines.append("-" * 84)
    lines.append(
        f"{'dtype':<10} {'shape':<18} {'I':>8} "
        f"{'ref_ms':>10} {'fused_ms':>10} {'speedup':>9}"
    )
    lines.append("-" * 84)

    for item in phase_results:
        lines.append(
            f"{item['dtype']:<10} {str(item['shape']):<18} "
            f"{item['inter_size']:>8} "
            f"{item['ref_median_ms']:>10.4f} "
            f"{item['fused_median_ms']:>10.4f} "
            f"{item['speedup']:>8.2f}x"
        )

    speedups = [float(r["speedup"]) for r in phase_results]
    lines.append(
        f"Summary: avg={statistics.mean(speedups):.2f}x, "
        f"min={min(speedups):.2f}x, max={max(speedups):.2f}x"
    )
    lines.append("")

    lines.append(f"{label} Error Metrics")
    lines.append("-" * 90)
    lines.append(
        f"{'dtype':<10} {'shape':<18} {'max_abs':>13} {'rmse':>13} "
        f"{'max_rel':>13} {'rel_l2':>13}"
    )
    lines.append("-" * 90)
    for item in phase_results:
        lines.append(
            f"{item['dtype']:<10} {str(item['shape']):<18} "
            f"{item['max_abs_err']:>13.4e} "
            f"{item['rmse']:>13.4e} "
            f"{item['max_rel_err']:>13.4e} "
            f"{item['rel_l2_err']:>13.4e}"
        )
    lines.append("")


def format_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "NovaDiT fused Gate-Act CUDA Benchmark Report",
        "=" * 48,
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

    append_phase_table(lines, "geglu", "GeGLU (fused_geglu vs geglu_ref)", results)
    append_phase_table(lines, "silu_gate", "SiLU-Gated (fused_gate_act silu vs gate_act_ref)", results)
    append_phase_table(lines, "gelu_gate", "GELU-Gated (fused_gate_act gelu vs gate_act_ref)", results)

    lines.extend([
        "Notes",
        "-" * 48,
        "GeGLU uses tanh-approximation GELU: 0.5*t*(1+tanh(sqrt(2/pi)*(t+0.044715*t^3))).",
        "SiLU-Gated uses y = z_up * (z_gate * sigmoid(z_gate)).",
        "Aligned vs dynamic path is selected based on inter_size (I).",
        "Latency values are CUDA event medians; allocations are included.",
    ])
    return "\n".join(lines)


# ---- Main ----

def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NovaDiT fused gate-act CUDA kernels."
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["geglu", "silu_gate", "gelu_gate", "all"],
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gate_act_benchmark_report.txt"),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["fp16", "bf16", "fp32"],
        choices=["fp16", "bf16", "fp32"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not HAS_NOVADIT_CUDA:
        raise RuntimeError("novadit_cuda extension is not built/importable.")

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

    # shapes as (B, S, 2*I)
    shapes: list[Shape] = [
        (8, 256, 2 * 768),    # I=768  aligned
        (4, 1024, 2 * 1152),  # I=1152 aligned
        (8, 256, 2 * 770),    # I=770  dynamic
    ]
    results: list[BenchmarkResult] = []

    for dtype_name in args.dtypes:
        dtype = parse_dtype(dtype_name)
        for shape in shapes:
            if args.mode in ("geglu", "all"):
                print(f"Benchmarking geglu dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_geglu_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )
            if args.mode in ("silu_gate", "all"):
                print(f"Benchmarking silu_gate dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_silu_case(
                        dtype=dtype, shape=shape,
                        warmup=args.warmup, iters=args.iters,
                    )
                )
            if args.mode in ("gelu_gate", "all"):
                print(f"Benchmarking gelu_gate dtype={dtype_name}, shape={shape} ...")
                results.append(
                    run_gelu_gate_case(
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
