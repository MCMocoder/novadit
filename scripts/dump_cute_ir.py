#!/usr/bin/env python3
"""Dump MLIR / PTX / SASS for a CuTe DSL compiled kernel.

Usage:
  python scripts/dump_cute_ir.py [--M 2048] [--H 768] [--I 3072] [--act silu]

Environment variables (optional):
  CUTE_DSL_DUMP_DIR   output directory (default: ./cute_dump)
  CUTE_DSL_KEEP       comma-separated artifacts: ir,ir-debug,ptx,cubin,all
                      (default: all)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "csrc" / "cute_dsl" / "fc1_bias_act"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump CuTe DSL IR/PTX/SASS")
    p.add_argument("--M", type=int, default=2048, help="token count (M)")
    p.add_argument("--H", type=int, default=768, help="hidden size (K)")
    p.add_argument("--I", type=int, default=3072, help="intermediate size (N)")
    p.add_argument("--act", choices=["silu", "gelu"], default="silu")
    p.add_argument("--tile", type=str, default="128,256,64",
                   help="tile shape M,N,K (default: 128,256,64)")
    p.add_argument("--dump-dir", type=Path, default=Path("cute_dump"),
                   help="output directory (default: cute_dump)")
    p.add_argument("--keep", type=str, default="all",
                   help="artifacts to keep: ir,ir-debug,ptx,cubin,all")
    p.add_argument("--no-sass", action="store_true",
                   help="skip SASS disassembly (requires cuobjdump)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = str(args.dump_dir.resolve())

    # Set env vars for the CUTLASS DSL compiler.
    os.environ["CUTE_DSL_DUMP_DIR"] = dump_dir
    os.environ["CUTE_DSL_KEEP"] = args.keep

    import cuda.bindings.driver  # noqa: F401 – initialise CUDA driver context
    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from novadit.ops import interleave_weight, split_gated_weight
    from fc1_bias_act import (
        ACT_GELU, ACT_SILU,
        FC1GatedActSM120BF16Kernel,
        FusedGatedMlpShape,
    )

    M, H, I = args.M, args.H, args.I
    tile = tuple(int(x) for x in args.tile.split(","))
    act_id = ACT_SILU if args.act == "silu" else ACT_GELU
    dtype = torch.float16

    # Build dummy input tensors for compilation.
    x_cpu = torch.empty(M, H, dtype=dtype)
    w_cpu = torch.empty(2 * I, H, dtype=dtype)
    w_inter = interleave_weight(w_cpu)
    w_gate_cpu, w_value_cpu = split_gated_weight(w_inter)
    y_cpu = torch.empty(M, I, dtype=dtype)

    x_cute = _make_cute_3d(x_cpu)
    wg_cute = _make_cute_3d(w_gate_cpu)
    wv_cute = _make_cute_3d(w_value_cpu)
    y_cute = _make_cute_3d(y_cpu)

    kernel = FC1GatedActSM120BF16Kernel(
        FusedGatedMlpShape(batch_tokens=M, hidden_size=H, intermediate_size=I),
        fc1_tile_shape_mnk=tile,
    )

    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    stream = ct.default_stream()

    print(f"Compiling CuTe DSL kernel (M={M}, H={H}, I={I}, tile={tile}, act={args.act}) …")
    compiled = cute.compile(
        kernel, x_cute, wg_cute, wv_cute, y_cute,
        act_id, max_clusters, stream,
    )

    print(f"Dumped artifacts → {dump_dir}/")
    _list_dir(dump_dir)

    # Try to find PTX and disassemble to SASS.
    if not args.no_sass:
        cubin_files = sorted(Path(dump_dir).glob("*.cubin"))
        for cf in cubin_files:
            sass_path = cf.with_suffix(".sass")
            _run(f"cuobjdump -sass {cf} > {sass_path}")
            if sass_path.exists() and sass_path.stat().st_size > 0:
                print(f"SASS written → {sass_path}")
            else:
                sass_path.unlink(missing_ok=True)

    print("Done.")


def _make_cute_3d(t: "torch.Tensor") -> "object":
    import cutlass
    from cutlass import Float16
    import cutlass.torch as ct
    cute_t, _ = ct.cute_tensor_like(
        t.unsqueeze(-1).cpu(), Float16, False, 16,
    )
    return cute_t


def _list_dir(path: str, indent: int = 2) -> None:
    """List files in dump directory sorted by size."""
    prefix = " " * indent
    entries = sorted(
        Path(path).iterdir(),
        key=lambda p: (-p.stat().st_size, p.name),
    )
    for e in entries:
        if e.is_file():
            size_kb = e.stat().st_size / 1024
            print(f"{prefix}{e.name}  ({size_kb:.1f} KB)")


def _run(cmd: str) -> None:
    try:
        subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    except Exception:
        pass


if __name__ == "__main__":
    main()
