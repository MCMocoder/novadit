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
    adaln_compile,
    adaln_ref,
    adarmsnorm_compile,
    adarmsnorm_ref,
    fused_adaln,
    fused_adarmsnorm,
    fused_adaln_inference,
    fused_adarmsnorm_inference,
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
    actual_float = actual.float()
    expected_float = expected.float()
    diff = actual_float - expected_float
    abs_diff = diff.abs()
    denominator = expected_float.abs().clamp_min(relative_error_floor(dtype))
    expected_l2 = torch.linalg.vector_norm(expected_float)

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
    compile_times: tuple[float, float, float],
    errors_fused: ErrorMetrics,
    errors_compile: ErrorMetrics,
) -> BenchmarkResult:
    ref_median, _, _ = ref_times
    fused_median, fused_mean, fused_p90 = fused_times
    compile_median, compile_mean, compile_p90 = compile_times
    return {
        "phase": phase,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": shape,
        "ref_median_ms": ref_median,
        "fused_median_ms": fused_median,
        "fused_mean_ms": fused_mean,
        "fused_p90_ms": fused_p90,
        "compile_median_ms": compile_median,
        "compile_mean_ms": compile_mean,
        "compile_p90_ms": compile_p90,
        "speedup_fused": ref_median / fused_median if fused_median > 0 else math.inf,
        "speedup_compile": ref_median / compile_median if compile_median > 0 else math.inf,
        "fused_vs_compile": fused_median / compile_median if compile_median > 0 else math.inf,
        "errors_fused": errors_fused,
        "errors_compile": errors_compile,
    }


# ---------------------------------------------------------------------------
# AdaLN benchmarks
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_adaln_forward(
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
        return adaln_ref(x, scale, shift, eps=1e-5)

    def run_fused() -> torch.Tensor:
        return fused_adaln(x, scale, shift, eps=1e-5)

    def run_compile() -> torch.Tensor:
        return adaln_compile(x, scale, shift, eps=1e-5)

    ref = run_ref()
    out_fused = run_fused()
    out_compile = run_compile()

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out_fused, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="forward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=error_metrics(out_fused, ref, dtype=dtype),
        errors_compile=error_metrics(out_compile, ref, dtype=dtype),
    )


@torch.no_grad()
def run_adaln_inference(
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
        return adaln_ref(x, scale, shift, eps=1e-5)

    def run_fused() -> torch.Tensor:
        return fused_adaln_inference(x, scale, shift, eps=1e-5)

    def run_compile() -> torch.Tensor:
        return adaln_compile(x, scale, shift, eps=1e-5)

    ref = run_ref()
    out_fused = run_fused()
    out_compile = run_compile()

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out_fused, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="inference",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=error_metrics(out_fused, ref, dtype=dtype),
        errors_compile=error_metrics(out_compile, ref, dtype=dtype),
    )


def run_adaln_full_chain(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)
    inputs = (x, scale, shift)

    ref_y = adaln_ref(x, scale, shift, eps=1e-5)
    fused_y = fused_adaln(x, scale, shift, eps=1e-5)
    compile_y = adaln_compile(x, scale, shift, eps=1e-5)

    ref_outputs = ref_y, *torch.autograd.grad(ref_y, inputs, dy, allow_unused=False)
    fused_outputs = fused_y, *torch.autograd.grad(fused_y, inputs, dy, allow_unused=False)
    compile_outputs = compile_y, *torch.autograd.grad(compile_y, inputs, dy, allow_unused=False)

    rtol, atol = tolerances(dtype)
    for fo, ro in zip(fused_outputs, ref_outputs, strict=True):
        torch.testing.assert_close(fo, ro, rtol=rtol, atol=atol)
    for co, ro in zip(compile_outputs, ref_outputs, strict=True):
        torch.testing.assert_close(co, ro, rtol=rtol, atol=atol)

    component_errors_fused = {
        name: error_metrics(fo, ro, dtype=dtype)
        for name, fo, ro in zip(("y", "dx", "dscale", "dshift"), fused_outputs, ref_outputs, strict=True)
    }
    component_errors_compile = {
        name: error_metrics(co, ro, dtype=dtype)
        for name, co, ro in zip(("y", "dx", "dscale", "dshift"), compile_outputs, ref_outputs, strict=True)
    }
    full_errors_fused = {
        metric: max(e[metric] for e in component_errors_fused.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    full_errors_compile = {
        metric: max(e[metric] for e in component_errors_compile.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }

    def _run_full(mode: str) -> tuple[torch.Tensor, ...]:
        x_i = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        scale_i = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        shift_i = torch.randn_like(scale_i, requires_grad=True)
        dy_i = torch.randn_like(x_i)
        inputs_i = (x_i, scale_i, shift_i)

        if mode == "ref":
            y = adaln_ref(x_i, scale_i, shift_i, eps=1e-5)
        elif mode == "fused":
            y = fused_adaln(x_i, scale_i, shift_i, eps=1e-5)
        else:
            y = adaln_compile(x_i, scale_i, shift_i, eps=1e-5)

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
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=full_errors_fused,
        errors_compile=full_errors_compile,
    )


def run_adaln_backward(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)
    inputs = (x, scale, shift)

    ref_y = adaln_ref(x, scale, shift, eps=1e-5)
    fused_y = fused_adaln(x, scale, shift, eps=1e-5)
    compile_y = adaln_compile(x, scale, shift, eps=1e-5)

    ref_grads = torch.autograd.grad(ref_y, inputs, dy, allow_unused=False)
    fused_grads = torch.autograd.grad(fused_y, inputs, dy, allow_unused=False)
    compile_grads = torch.autograd.grad(compile_y, inputs, dy, allow_unused=False)

    rtol, atol = tolerances(dtype)
    for fg, rg in zip(fused_grads, ref_grads, strict=True):
        torch.testing.assert_close(fg, rg, rtol=rtol, atol=atol)
    for cg, rg in zip(compile_grads, ref_grads, strict=True):
        torch.testing.assert_close(cg, rg, rtol=rtol, atol=atol)

    component_errors_fused = {
        name: error_metrics(fg, rg, dtype=dtype)
        for name, fg, rg in zip(("dx", "dscale", "dshift"), fused_grads, ref_grads, strict=True)
    }
    component_errors_compile = {
        name: error_metrics(cg, rg, dtype=dtype)
        for name, cg, rg in zip(("dx", "dscale", "dshift"), compile_grads, ref_grads, strict=True)
    }
    full_errors_fused = {
        metric: max(e[metric] for e in component_errors_fused.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    full_errors_compile = {
        metric: max(e[metric] for e in component_errors_compile.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }

    def _run_bwd(mode: str) -> tuple[torch.Tensor, ...]:
        x_i = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        scale_i = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        shift_i = torch.randn_like(scale_i, requires_grad=True)
        dy_i = torch.randn_like(x_i)
        inputs_i = (x_i, scale_i, shift_i)

        if mode == "ref":
            y = adaln_ref(x_i, scale_i, shift_i, eps=1e-5)
        elif mode == "fused":
            y = fused_adaln(x_i, scale_i, shift_i, eps=1e-5)
        else:
            y = adaln_compile(x_i, scale_i, shift_i, eps=1e-5)

        return torch.autograd.grad(y, inputs_i, dy_i, allow_unused=False)

    return make_result(
        phase="backward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(lambda: _run_bwd("ref"), warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(lambda: _run_bwd("fused"), warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(lambda: _run_bwd("compile"), warmup=warmup, iters=iters),
        errors_fused=full_errors_fused,
        errors_compile=full_errors_compile,
    )


# ---------------------------------------------------------------------------
# AdaRMSNorm benchmarks
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_adarmsnorm_forward(
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
    out_fused = run_fused()
    out_compile = run_compile()

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out_fused, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="forward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=error_metrics(out_fused, ref, dtype=dtype),
        errors_compile=error_metrics(out_compile, ref, dtype=dtype),
    )


@torch.no_grad()
def run_adarmsnorm_inference(
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
    out_fused = run_fused()
    out_compile = run_compile()

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(out_fused, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_compile, ref, rtol=rtol, atol=atol)

    return make_result(
        phase="inference",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(run_ref, warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(run_fused, warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=error_metrics(out_fused, ref, dtype=dtype),
        errors_compile=error_metrics(out_compile, ref, dtype=dtype),
    )


def run_adarmsnorm_full_chain(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
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

    component_errors_fused = {
        name: error_metrics(fo, ro, dtype=dtype)
        for name, fo, ro in zip(("y", "dx", "dscale", "dshift"), fused_outputs, ref_outputs, strict=True)
    }
    component_errors_compile = {
        name: error_metrics(co, ro, dtype=dtype)
        for name, co, ro in zip(("y", "dx", "dscale", "dshift"), compile_outputs, ref_outputs, strict=True)
    }
    full_errors_fused = {
        metric: max(e[metric] for e in component_errors_fused.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    full_errors_compile = {
        metric: max(e[metric] for e in component_errors_compile.values())
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
        compile_times=cuda_event_benchmark(run_compile, warmup=warmup, iters=iters),
        errors_fused=full_errors_fused,
        errors_compile=full_errors_compile,
    )


def run_adarmsnorm_backward(
    *,
    dtype: torch.dtype,
    shape: Shape,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    batch_size, _, hidden_size = shape

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)
    inputs = (x, scale, shift)

    ref_y = adarmsnorm_ref(x, scale, shift, eps=1e-5)
    fused_y = fused_adarmsnorm(x, scale, shift, eps=1e-5)
    compile_y = adarmsnorm_compile(x, scale, shift, eps=1e-5)

    ref_grads = torch.autograd.grad(ref_y, inputs, dy, allow_unused=False)
    fused_grads = torch.autograd.grad(fused_y, inputs, dy, allow_unused=False)
    compile_grads = torch.autograd.grad(compile_y, inputs, dy, allow_unused=False)

    rtol, atol = tolerances(dtype)
    for fg, rg in zip(fused_grads, ref_grads, strict=True):
        torch.testing.assert_close(fg, rg, rtol=rtol, atol=atol)
    for cg, rg in zip(compile_grads, ref_grads, strict=True):
        torch.testing.assert_close(cg, rg, rtol=rtol, atol=atol)

    component_errors_fused = {
        name: error_metrics(fg, rg, dtype=dtype)
        for name, fg, rg in zip(("dx", "dscale", "dshift"), fused_grads, ref_grads, strict=True)
    }
    component_errors_compile = {
        name: error_metrics(cg, rg, dtype=dtype)
        for name, cg, rg in zip(("dx", "dscale", "dshift"), compile_grads, ref_grads, strict=True)
    }
    full_errors_fused = {
        metric: max(e[metric] for e in component_errors_fused.values())
        for metric in ("max_abs_err", "rmse", "max_rel_err", "rel_l2_err")
    }
    full_errors_compile = {
        metric: max(e[metric] for e in component_errors_compile.values())
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

    return make_result(
        phase="backward",
        dtype=dtype,
        shape=shape,
        ref_times=cuda_event_benchmark(lambda: _run_bwd("ref"), warmup=warmup, iters=iters),
        fused_times=cuda_event_benchmark(lambda: _run_bwd("fused"), warmup=warmup, iters=iters),
        compile_times=cuda_event_benchmark(lambda: _run_bwd("compile"), warmup=warmup, iters=iters),
        errors_fused=full_errors_fused,
        errors_compile=full_errors_compile,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(kernel_name: str, results: list[BenchmarkResult]) -> str:
    lines = [
        f"NovaDiT fused {kernel_name} Benchmark: ref vs CUDA vs torch.compile",
        "=" * 90,
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

    for phase in ("inference", "forward", "full_chain", "backward"):
        phase_results = [r for r in results if r["phase"] == phase]
        if not phase_results:
            continue

        lines.append(phase.replace("_", " ").title())
        lines.append("-" * 90)
        lines.append(
            f"{'dtype':<10} {'shape':<18} "
            f"{'ref_ms':>9} {'fused_ms':>9} {'compile_ms':>10} "
            f"{'f_speedup':>9} {'c_speedup':>9} {'f_vs_c':>7}"
        )
        lines.append("-" * 90)

        for item in phase_results:
            lines.append(
                f"{item['dtype']:<10} {str(item['shape']):<18} "
                f"{item['ref_median_ms']:>9.4f} "
                f"{item['fused_median_ms']:>9.4f} "
                f"{item['compile_median_ms']:>10.4f} "
                f"{item['speedup_fused']:>8.2f}x "
                f"{item['speedup_compile']:>8.2f}x "
                f"{item['fused_vs_compile']:>6.2f}x"
            )

        fused_speedups = [float(r["speedup_fused"]) for r in phase_results]
        compile_speedups = [float(r["speedup_compile"]) for r in phase_results]
        lines.append(
            f"fused avg={statistics.mean(fused_speedups):.2f}x "
            f"compile avg={statistics.mean(compile_speedups):.2f}x"
        )
        lines.append("")

        lines.append(f"{phase.replace('_', ' ').title()} Error Metrics")
        lines.append("-" * 120)
        lines.append(
            f"{'dtype':<10} {'shape':<18} {'type':<6} "
            f"{'max_abs':>13} {'rmse':>13} {'max_rel':>13} {'rel_l2':>13}"
        )
        lines.append("-" * 120)
        for item in phase_results:
            for prefix, errors in [("fused", item["errors_fused"]), ("comp", item["errors_compile"])]:
                lines.append(
                    f"{item['dtype']:<10} {str(item['shape']):<18} {prefix:<6} "
                    f"{errors['max_abs_err']:>13.4e} "
                    f"{errors['rmse']:>13.4e} "
                    f"{errors['max_rel_err']:>13.4e} "
                    f"{errors['rel_l2_err']:>13.4e}"
                )
        lines.append("")

    lines.extend([
        "Notes",
        "-" * 48,
        "ref: pure PyTorch eager-mode reference implementation.",
        "fused: hand-written CUDA kernel via novadit_cuda extension.",
        "compile: torch.compile auto-fused Triton kernel baseline.",
        "f_speedup = ref_ms / fused_ms, c_speedup = ref_ms / compile_ms.",
        "f_vs_c = fused_ms / compile_ms (>1 means compile is faster).",
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
        description="Benchmark torch.compile baseline vs hand-written CUDA fused norm kernels."
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--kernel",
        choices=["adaln", "adarmsnorm", "all"],
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "forward", "full_chain", "backward", "all"],
        default="all",
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

    for kernel_name in ("adaln", "adarmsnorm"):
        if args.kernel not in (kernel_name, "all"):
            continue

        results: list[BenchmarkResult] = []

        for dtype_name in args.dtypes:
            dtype = parse_dtype(dtype_name)
            for shape in shapes:
                if kernel_name == "adaln":
                    forward_fn = run_adaln_forward
                    inference_fn = run_adaln_inference
                    full_chain_fn = run_adaln_full_chain
                    backward_fn = run_adaln_backward
                else:
                    forward_fn = run_adarmsnorm_forward
                    inference_fn = run_adarmsnorm_inference
                    full_chain_fn = run_adarmsnorm_full_chain
                    backward_fn = run_adarmsnorm_backward

                if args.mode in ("inference", "all"):
                    print(f"[{kernel_name}] inference {dtype_name} {shape} ...")
                    results.append(inference_fn(dtype=dtype, shape=shape, warmup=args.warmup, iters=args.iters))
                if args.mode in ("forward", "all"):
                    print(f"[{kernel_name}] forward {dtype_name} {shape} ...")
                    results.append(forward_fn(dtype=dtype, shape=shape, warmup=args.warmup, iters=args.iters))
                if args.mode in ("full_chain", "all"):
                    print(f"[{kernel_name}] full_chain {dtype_name} {shape} ...")
                    results.append(full_chain_fn(dtype=dtype, shape=shape, warmup=args.warmup, iters=args.iters))
                if args.mode in ("backward", "all"):
                    print(f"[{kernel_name}] backward {dtype_name} {shape} ...")
                    results.append(backward_fn(dtype=dtype, shape=shape, warmup=args.warmup, iters=args.iters))

        report = format_report(kernel_name, results)
        out_path = Path(f"norm_compile_{kernel_name}_benchmark.txt")
        out_path.write_text(report, encoding="utf-8")
        print("")
        print(report)
        print(f"TXT report written to: {out_path}")


if __name__ == "__main__":
    main()
