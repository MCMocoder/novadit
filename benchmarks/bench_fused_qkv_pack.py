from __future__ import annotations

import time

import torch

from novadit.ops import fused_qkv_pack, qkv_pack_ref


def bench(fn, qkv) -> float:
    start = time.perf_counter()
    _ = fn(qkv, num_heads=8, head_dim=64)
    if qkv.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qkv = torch.randn(2, 1024, 3 * 512, device=device)
    print(f"qkv_pack_ref_ms={bench(qkv_pack_ref, qkv):.3f}")
    print(f"fused_qkv_pack_ms={bench(fused_qkv_pack, qkv):.3f}")


if __name__ == "__main__":
    main()
