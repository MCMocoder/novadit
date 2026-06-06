from __future__ import annotations

import torch

from novadit.modules import NovaDiTAttention


def main() -> None:
    x = torch.randn(2, 16, 64)
    module = NovaDiTAttention(hidden_size=64, num_heads=4)
    y = module(x)
    print("attention output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
