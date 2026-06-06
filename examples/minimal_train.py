from __future__ import annotations

import torch

from novadit.modules import NovaDiTBlock


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = NovaDiTBlock(hidden_size=64, num_heads=4).to(device=device, dtype=dtype)
    x = torch.randn(2, 16, 64, device=device, dtype=dtype, requires_grad=True)
    attn_scale = torch.randn(2, 64, device=device, dtype=dtype)
    attn_shift = torch.randn(2, 64, device=device, dtype=dtype)
    mlp_scale = torch.randn(2, 64, device=device, dtype=dtype)
    mlp_shift = torch.randn(2, 64, device=device, dtype=dtype)

    out = model(x, attn_scale, attn_shift, mlp_scale, mlp_shift)
    loss = out.float().pow(2).mean()
    loss.backward()
    print("minimal_train complete", loss.item())


if __name__ == "__main__":
    main()
