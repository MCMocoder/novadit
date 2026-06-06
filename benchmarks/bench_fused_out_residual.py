from __future__ import annotations

import time

import torch

from novadit.ops import fused_out_residual, out_residual_ref


def bench(fn, x, residual, bias) -> float:
    start = time.perf_counter()
    _ = fn(x, residual, bias)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 1024, 512, device=device)
    residual = torch.randn_like(x)
    bias = torch.randn(512, device=device)
    print(f"out_residual_ref_ms={bench(out_residual_ref, x, residual, bias):.3f}")
    print(f"fused_out_residual_ms={bench(fused_out_residual, x, residual, bias):.3f}")


if __name__ == "__main__":
    main()
