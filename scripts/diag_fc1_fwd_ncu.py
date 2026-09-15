"""
Forward-only CuTe DSL FC1 kernel diagnostic for Nsight Compute GUI.

Generates a single, pause-able kernel launch suitable for ncu-gui attach.
Prints the kernel name so you can filter in the ncu-gui "API Stream" view.

Usage:
  # Terminal 1: run the script (it pauses before launching kernel)
  python scripts/diag_fc1_fwd_ncu.py --shape 2,128,768,768 --dtype fp16 --clusters 1

  # Terminal 2: attach ncu-gui
  ncu-ui --target-process $(pgrep -f diag_fc1_fwd_ncu)

  # Or collect a report file then open in ncu-gui:
  ncu --set full -o /tmp/fc1_fwd_profile \
      python scripts/diag_fc1_fwd_ncu.py --shape 2,128,768,768 --dtype fp16 --no-pause

  ncu-ui /tmp/fc1_fwd_profile.ncu-rep
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "csrc" / "cute_dsl" / "fc1_bias_act"))

import cuda.bindings.driver  # noqa: F401 — must be before cutlass
import torch


def build_and_run(
    M: int, H: int, I: int, act: str, dtype: torch.dtype, max_clusters: int,
) -> None:
    """Compile the forward kernel once and launch it ONCE for profiling."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from fc1_bias_act import FC1GatedActSM120BF16Kernel, FusedGatedMlpShape

    cute_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    scale = 1.0 / (H ** 0.5)

    # ── compile ──────────────────────────────────────────────────────
    def _empty(m, n):
        t = torch.empty(m, n, dtype=dtype)
        cute_t, _ = ct.cute_tensor_like(t.unsqueeze(-1).cpu(), cute_dtype, False, 16)
        return cute_t

    kernel = FC1GatedActSM120BF16Kernel(
        FusedGatedMlpShape(batch_tokens=M, hidden_size=H, intermediate_size=I),
        ab_dtype=cute_dtype, output_dtype=cute_dtype,
    )
    hw = cutlass.utils.HardwareInfo()
    stream = ct.default_stream()
    act_id = 0 if act == "silu" else 1

    compiled = cute.compile(
        kernel,
        _empty(M, H), _empty(I, H), _empty(I, H), _empty(M, I),
        act_id, max_clusters, stream,
    )

    # ── tensors ──────────────────────────────────────────────────────
    x = torch.randn(M, H, device="cuda", dtype=dtype) * scale
    w = torch.randn(2 * I, H, device="cuda", dtype=dtype) * scale
    wg, wv = w[:I].contiguous(), w[I:].contiguous()
    y = torch.empty(M, I, device="cuda", dtype=dtype)

    def _wrap(t):
        t3d = t.unsqueeze(-1).contiguous()
        return ct.from_dlpack(t3d, assumed_align=16), t3d

    xc, _ = _wrap(x)
    wgc, _ = _wrap(wg)
    wvc, _ = _wrap(wv)
    yc, y3d = _wrap(y)

    tiles_fwd = (M // 64) * (I // 64)
    print(f"\n  Shape:  B·S=M={M}  H={H}  I={I}")
    print(f"  Tiles:  {tiles_fwd} work tiles  |  K-tile cnt = {H // 64}  |  max_clusters = {max_clusters}")
    print(f"  SM:     {torch.cuda.get_device_capability(0)}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  dtype:  {dtype}  act: {act}")
    print()

    # ── launch ───────────────────────────────────────────────────────
    print("  Launching kernel ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled(xc, wgc, wvc, yc, stream)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Kernel wall-clock: {elapsed_ms:.3f} ms  ({elapsed_ms / tiles_fwd:.1f} us/tile)")
    print()
    print("  Done.  Kernel completed successfully.")
    print("  (If profiling, the kernel launch is captured at the barrier above.)")


def main():
    parser = argparse.ArgumentParser(
        description="Single-launch CuTe DSL forward kernel for Nsight Compute GUI"
    )
    parser.add_argument(
        "--shape", default="2,128,768,768",
        help="B,S,H,I (default: 2,128,768,768)",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--act", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--clusters", type=int, default=16,
                        help="max_active_clusters (default: 16; try 1 vs 32 vs sm_count)")
    parser.add_argument("--no-pause", action="store_true",
                        help="Skip pause (for ncu CLI collection)")
    parser.add_argument("--pause-sec", type=int, default=0,
                        help="Extra pause before launch (seconds)")
    args = parser.parse_args()

    parts = args.shape.split(",")
    B, S, H, I = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    M = B * S
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print("=" * 70)
    print("  CuTe DSL Forward Kernel — Nsight Compute Diagnostic")
    print("=" * 70)
    print(f"  PID: {os.getpid()}")
    print()

    if not args.no_pause:
        if args.pause_sec:
            print(f"  Pausing {args.pause_sec}s for ncu-gui attach ...")
            time.sleep(args.pause_sec)
        input("  Press ENTER to compile and launch the kernel ...")

    build_and_run(M, H, I, args.act, dtype, args.clusters)


if __name__ == "__main__":
    main()
