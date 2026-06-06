from __future__ import annotations

import torch

from novadit.modules import NovaDiTBlock


def main() -> None:
    x = torch.randn(2, 16, 64)
    scale = torch.randn(2, 64)
    shift = torch.randn(2, 64)
    module = NovaDiTBlock(hidden_size=64, num_heads=4)
    y = module(x, scale, shift, scale, shift)
    print("block output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
