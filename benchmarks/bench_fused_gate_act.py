from __future__ import annotations

import time

import torch

from novadit.ops import fused_gate_act, gate_act_ref


def bench(fn, x) -> float:
    start = time.perf_counter()
    _ = fn(x, activation="silu")
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 1024, 2 * 1024, device=device)
    print(f"gate_act_ref_ms={bench(gate_act_ref, x):.3f}")
    print(f"fused_gate_act_ms={bench(fused_gate_act, x):.3f}")


if __name__ == "__main__":
    main()
