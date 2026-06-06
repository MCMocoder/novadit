# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from novadit.modules import NovaDiTBlock
from novadit.ops import HAS_NOVADIT_CUDA


class DiTStack(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        use_fused_adaln: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NovaDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_p=0.0,
                    attention_backend="sdpa",
                    use_fused_adaln=use_fused_adaln,
                    # These kernels do not have a complete training path yet.
                    use_fused_qkv_pack=False,
                    use_fused_out_residual=False,
                    use_fused_gate_act=False,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, *modulations: torch.Tensor) -> torch.Tensor:
        expected = 4 * len(self.blocks)
        if len(modulations) != expected:
            raise ValueError(f"Expected {expected} modulation tensors")

        for layer_idx, block in enumerate(self.blocks):
            offset = 4 * layer_idx
            x = block(x, *modulations[offset : offset + 4])
        return x


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
    samples: list[float] = []

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(len(ordered) * 0.9))
    return statistics.median(samples), statistics.mean(samples), ordered[p90_index]


def dtype_tolerances(dtype: torch.dtype, depth: int) -> tuple[float, float]:
    depth_scale = max(1.0, math.sqrt(depth))
    if dtype == torch.float32:
        return 5e-3 * depth_scale, 5e-3 * depth_scale
    if dtype == torch.float16:
        return 1e-1 * depth_scale, 1e-1 * depth_scale
    if dtype == torch.bfloat16:
        return 2e-1 * depth_scale, 2e-1 * depth_scale
    raise ValueError(f"Unsupported dtype: {dtype}")


def gradient_rel_l2_tolerance(dtype: torch.dtype, depth: int) -> float:
    depth_scale = max(1.0, math.sqrt(depth))
    if dtype == torch.float32:
        return 1e-5 * depth_scale
    if dtype == torch.float16:
        return 3e-3 * depth_scale
    if dtype == torch.bfloat16:
        return 2e-2 * depth_scale
    raise ValueError(f"Unsupported dtype: {dtype}")


def group_error_metrics(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> tuple[float, float]:
    max_abs = 0.0
    diff_l2_sq = 0.0
    expected_l2_sq = 0.0

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        diff = actual_tensor.float() - expected_tensor.float()
        max_abs = max(max_abs, diff.abs().max().item())
        diff_l2_sq += diff.square().sum().item()
        expected_l2_sq += expected_tensor.float().square().sum().item()

    rel_l2 = math.sqrt(diff_l2_sq / max(expected_l2_sq, 1e-24))
    return max_abs, rel_l2


def make_modulations(
    *,
    batch_size: int,
    hidden_size: int,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.randn(
            batch_size,
            hidden_size,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for _ in range(4 * depth)
    )


def clone_inputs(
    tensors: Sequence[torch.Tensor],
    *,
    requires_grad: bool,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.detach().clone().requires_grad_(requires_grad) for tensor in tensors
    )


def benchmark_dtype(
    *,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    depth: int,
    mlp_ratio: float,
    mode: str,
    warmup: int,
    iters: int,
) -> list[dict[str, float | str]]:
    device = torch.device("cuda")
    ref_model = DiTStack(
        hidden_size=hidden_size,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        use_fused_adaln=False,
        device=device,
        dtype=dtype,
    ).eval()
    fused_model = DiTStack(
        hidden_size=hidden_size,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        use_fused_adaln=True,
        device=device,
        dtype=dtype,
    ).eval()
    fused_model.load_state_dict(ref_model.state_dict())

    base_x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    base_modulations = make_modulations(
        batch_size=batch_size,
        hidden_size=hidden_size,
        depth=depth,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    rtol, atol = dtype_tolerances(dtype, depth)
    results: list[dict[str, float | str]] = []

    if mode in ("forward", "all"):
        with torch.no_grad():

            def run_ref_forward() -> torch.Tensor:
                return ref_model(base_x, *base_modulations)

            def run_fused_forward() -> torch.Tensor:
                return fused_model(base_x, *base_modulations)

            ref_y = run_ref_forward()
            fused_y = run_fused_forward()
            torch.testing.assert_close(fused_y, ref_y, rtol=rtol, atol=atol)
            output_max_abs, output_rel_l2 = group_error_metrics(
                (fused_y,), (ref_y,)
            )
            ref_times = cuda_event_benchmark(
                run_ref_forward, warmup=warmup, iters=iters
            )
            fused_times = cuda_event_benchmark(
                run_fused_forward, warmup=warmup, iters=iters
            )
            results.append(
                make_result(
                    phase="forward",
                    dtype=dtype,
                    ref_times=ref_times,
                    fused_times=fused_times,
                    output_max_abs=output_max_abs,
                    output_rel_l2=output_rel_l2,
                    grad_max_abs=0.0,
                    grad_rel_l2=0.0,
                )
            )

    if mode in ("full_chain", "all"):
        ref_inputs = clone_inputs((base_x, *base_modulations), requires_grad=True)
        fused_inputs = clone_inputs((base_x, *base_modulations), requires_grad=True)
        ref_targets = (*ref_inputs, *tuple(ref_model.parameters()))
        fused_targets = (*fused_inputs, *tuple(fused_model.parameters()))
        dy = torch.randn_like(base_x)

        def run_ref_full_chain() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            y = ref_model(*ref_inputs)
            grads = torch.autograd.grad(y, ref_targets, dy, allow_unused=False)
            return y, grads

        def run_fused_full_chain() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            y = fused_model(*fused_inputs)
            grads = torch.autograd.grad(y, fused_targets, dy, allow_unused=False)
            return y, grads

        ref_y, ref_grads = run_ref_full_chain()
        fused_y, fused_grads = run_fused_full_chain()
        torch.testing.assert_close(fused_y, ref_y, rtol=rtol, atol=atol)

        output_max_abs, output_rel_l2 = group_error_metrics((fused_y,), (ref_y,))
        grad_max_abs, grad_rel_l2 = group_error_metrics(fused_grads, ref_grads)
        if not math.isfinite(grad_rel_l2):
            raise AssertionError("Gradient relative L2 is not finite")
        grad_tolerance = gradient_rel_l2_tolerance(dtype, depth)
        if grad_rel_l2 > grad_tolerance:
            raise AssertionError(
                f"Gradient relative L2 {grad_rel_l2:.4e} exceeds "
                f"tolerance {grad_tolerance:.4e}"
            )
        ref_times = cuda_event_benchmark(
            run_ref_full_chain, warmup=warmup, iters=iters
        )
        fused_times = cuda_event_benchmark(
            run_fused_full_chain, warmup=warmup, iters=iters
        )
        results.append(
            make_result(
                phase="full_chain",
                dtype=dtype,
                ref_times=ref_times,
                fused_times=fused_times,
                output_max_abs=output_max_abs,
                output_rel_l2=output_rel_l2,
                grad_max_abs=grad_max_abs,
                grad_rel_l2=grad_rel_l2,
            )
        )

    return results


def make_result(
    *,
    phase: str,
    dtype: torch.dtype,
    ref_times: tuple[float, float, float],
    fused_times: tuple[float, float, float],
    output_max_abs: float,
    output_rel_l2: float,
    grad_max_abs: float,
    grad_rel_l2: float,
) -> dict[str, float | str]:
    ref_median, ref_mean, ref_p90 = ref_times
    fused_median, fused_mean, fused_p90 = fused_times
    return {
        "phase": phase,
        "dtype": str(dtype).replace("torch.", ""),
        "ref_median_ms": ref_median,
        "ref_mean_ms": ref_mean,
        "ref_p90_ms": ref_p90,
        "fused_median_ms": fused_median,
        "fused_mean_ms": fused_mean,
        "fused_p90_ms": fused_p90,
        "speedup": ref_median / fused_median,
        "output_max_abs": output_max_abs,
        "output_rel_l2": output_rel_l2,
        "grad_max_abs": grad_max_abs,
        "grad_rel_l2": grad_rel_l2,
    }


def format_report(
    results: list[dict[str, float | str]],
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    depth: int,
    mlp_ratio: float,
    warmup: int,
    iters: int,
) -> str:
    lines = [
        "NovaDiT Full-Chain Benchmark Report",
        "=" * 48,
        f"Generated at       : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"PyTorch version    : {torch.__version__}",
        f"CUDA version       : {torch.version.cuda}",
        f"GPU                : {torch.cuda.get_device_name(0)}",
        f"Device capability  : {torch.cuda.get_device_capability(0)}",
        "",
        "Configuration",
        "-" * 48,
        f"Shape              : ({batch_size}, {seq_len}, {hidden_size})",
        f"Attention heads    : {num_heads}",
        f"Depth              : {depth}",
        f"MLP ratio          : {mlp_ratio:g}",
        f"Warmup / iterations: {warmup} / {iters}",
        "Attention backend  : PyTorch SDPA",
        "Fused path         : AdaLN forward + backward",
        "Reference path     : PyTorch AdaLN formula",
        "",
    ]

    for phase in ("forward", "full_chain"):
        phase_results = [result for result in results if result["phase"] == phase]
        if not phase_results:
            continue
        title = phase.replace("_", " ").title()
        lines.extend(
            [
                title,
                "-" * 72,
                f"{'dtype':<10} {'ref_ms':>10} {'fused_ms':>10} "
                f"{'speedup':>9} {'out_abs':>12} {'out_l2':>10} "
                f"{'grad_abs':>12} {'grad_l2':>10}",
                "-" * 96,
            ]
        )
        for result in phase_results:
            lines.append(
                f"{result['dtype']:<10} "
                f"{float(result['ref_median_ms']):>10.4f} "
                f"{float(result['fused_median_ms']):>10.4f} "
                f"{float(result['speedup']):>8.2f}x "
                f"{float(result['output_max_abs']):>12.4e} "
                f"{float(result['output_rel_l2']):>10.3e} "
                f"{float(result['grad_max_abs']):>12.4e} "
                f"{float(result['grad_rel_l2']):>10.3e}"
            )
        speedups = [float(result["speedup"]) for result in phase_results]
        lines.append(
            f"Summary: avg={statistics.mean(speedups):.2f}x, "
            f"min={min(speedups):.2f}x, max={max(speedups):.2f}x"
        )
        lines.append("")

    lines.extend(
        [
            "Notes",
            "-" * 48,
            "Full chain includes DiT forward and gradients for x, all modulation",
            "inputs, and all Linear/attention/MLP parameters.",
            "Each timed full-chain iteration rebuilds the autograd graph.",
            "Latency values are CUDA event medians; allocations are included.",
            "QKV pack and gate activation stay on PyTorch until their backward",
            "kernels are implemented; the report does not claim those fusions.",
        ]
    )
    return "\n".join(lines)


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
        description="Benchmark a full NovaDiT block/stack forward and backward."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["forward", "full_chain", "all"],
        default="all",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["fp16", "bf16", "fp32"],
        default=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dit_full_chain_benchmark_report.txt"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not HAS_NOVADIT_CUDA:
        raise RuntimeError("novadit_cuda extension is not built/importable")
    if args.hidden % args.heads != 0:
        raise ValueError("--hidden must be divisible by --heads")
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.empty((1, 1), device="cuda") @ torch.empty((1, 1), device="cuda")
    torch.cuda.synchronize()

    results: list[dict[str, float | str]] = []
    for dtype_name in args.dtypes:
        dtype = parse_dtype(dtype_name)
        print(
            f"Benchmarking DiT dtype={dtype_name}, "
            f"shape=({args.batch}, {args.seq}, {args.hidden}), "
            f"depth={args.depth}, mode={args.mode} ..."
        )
        results.extend(
            benchmark_dtype(
                dtype=dtype,
                batch_size=args.batch,
                seq_len=args.seq,
                hidden_size=args.hidden,
                num_heads=args.heads,
                depth=args.depth,
                mlp_ratio=args.mlp_ratio,
                mode=args.mode,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    report = format_report(
        results,
        batch_size=args.batch,
        seq_len=args.seq,
        hidden_size=args.hidden,
        num_heads=args.heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        warmup=args.warmup,
        iters=args.iters,
    )
    args.out.write_text(report, encoding="utf-8")
    print("")
    print(report)
    print("")
    print(f"TXT report written to: {args.out}")


if __name__ == "__main__":
    main()
