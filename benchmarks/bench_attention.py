from __future__ import annotations

import time

import torch

from novadit.modules import NovaDiTAttention


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 256, 512, device=device)
    module = NovaDiTAttention(hidden_size=512, num_heads=8, attention_backend="auto").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        _ = module(x)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"attention device={device} latency_ms={elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
