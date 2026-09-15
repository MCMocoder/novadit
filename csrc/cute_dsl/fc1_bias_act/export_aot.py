from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _add_cutlass_dsl_to_path(repo_root: Path) -> None:
    cutlass_root = Path(
        os.environ.get("NOVADIT_CUTLASS_PATH", repo_root / "third_party" / "cutlass")
    )
    cute_dsl = cutlass_root / "python" / "CuTeDSL"
    if cute_dsl.is_dir():
        sys.path.insert(0, str(cute_dsl))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the NovaDiT fused gated FC1 + activation CuTe DSL kernel."
    )
    parser.add_argument("--batch-tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument(
        "--activation",
        choices=["silu", "gelu"],
        default="silu",
        help="Gated activation to bake into the AOT artifact (act_id is constexpr).",
    )
    parser.add_argument(
        "--max-active-clusters",
        type=int,
        default=1,
        help=(
            "max_active_clusters baked into the persistent grid (constexpr). "
            "Defaults to 1 (single wave, always correct). Tune to the target "
            "device SM count for throughput."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    parser.add_argument("--file-name", default="fc1_bias_act_aot")
    parser.add_argument("--function-prefix", default="novadit_fc1_bias_act")
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    _add_cutlass_dsl_to_path(repo_root)

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    if __package__:
        from .cuda_compat import cuda
        from .fc1_bias_act import (
            ACT_GELU,
            ACT_SILU,
            FC1GatedActSM120BF16Kernel,
            FusedGatedMlpShape,
        )
    else:
        from cuda_compat import cuda

        from fc1_bias_act import (
            ACT_GELU,
            ACT_SILU,
            FC1GatedActSM120BF16Kernel,
            FusedGatedMlpShape,
        )

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = args.batch_tokens
    h = args.hidden_size
    n = args.intermediate_size
    act_id = ACT_SILU if args.activation == "silu" else ACT_GELU

    # Build fake CuTe tensors for AOT compilation. All operands are 3D with a
    # static batch dimension L=1. stride_order=(1, 0, 2) makes the contraction
    # dim (H for A/B, N for C) the leading/contiguous dimension, i.e. row-major
    # (K-major A/B, N-major C) -- matching the kernel's validated layouts.
    x = make_fake_compact_tensor(
        cutlass.Float16, (m, h, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    w_gate = make_fake_compact_tensor(
        cutlass.Float16, (n, h, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    w_value = make_fake_compact_tensor(
        cutlass.Float16, (n, h, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    y = make_fake_compact_tensor(
        cutlass.Float16, (m, n, 1), stride_order=(1, 0, 2), assumed_align=16
    )

    kernel = FC1GatedActSM120BF16Kernel(
        FusedGatedMlpShape(
            batch_tokens=m, hidden_size=h, intermediate_size=n
        )
    )

    stream = cuda.CUstream(0)
    compiled = cute.compile(
        kernel,
        x,
        w_gate,
        w_value,
        y,
        act_id,
        args.max_active_clusters,
        stream,
    )
    compiled.export_to_c(
        file_path=str(args.out_dir),
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )


if __name__ == "__main__":
    main()
