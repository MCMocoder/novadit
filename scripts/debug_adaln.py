from __future__ import annotations

import argparse

import torch

from novadit.ops import (
    HAS_NOVADIT_CUDA,
    NOVADIT_CUDA_IMPORT_ERROR,
    fused_adaln,
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def run_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push("adaln_forward")
    try:
        return fused_adaln(x, scale, shift)
    finally:
        torch.cuda.nvtx.range_pop()


def run_backward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    y = run_forward(x, scale, shift)
    torch.cuda.nvtx.range_push("adaln_backward")
    try:
        return torch.autograd.grad(y, (x, scale, shift), dy)
    finally:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Small AdaLN CUDA debug target.")
    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "all"],
        default="backward",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp32",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not HAS_NOVADIT_CUDA:
        raise RuntimeError(
            f"novadit_cuda extension is not importable: {NOVADIT_CUDA_IMPORT_ERROR!r}"
        )

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

    dtype = parse_dtype(args.dtype)
    shape = (args.batch, args.seq, args.hidden)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(
        args.batch,
        args.hidden,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    shift = torch.randn_like(scale, requires_grad=True)
    dy = torch.randn_like(x)

    for _ in range(args.iters):
        if args.mode == "forward":
            run_forward(x, scale, shift)
        elif args.mode == "backward":
            run_backward(x, scale, shift, dy)
        else:
            run_forward(x, scale, shift)
            run_backward(x, scale, shift, dy)

    torch.cuda.synchronize()
    print(
        f"AdaLN debug run complete: mode={args.mode}, dtype={args.dtype}, "
        f"shape={shape}, iters={args.iters}"
    )


if __name__ == "__main__":
    main()
