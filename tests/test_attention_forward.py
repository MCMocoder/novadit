from __future__ import annotations

from tests.helpers import requires_torch


def test_attention_forward_shape() -> None:
    requires_torch()

    import torch

    from novadit.modules import NovaDiTAttention

    x = torch.randn(2, 8, 32)
    module = NovaDiTAttention(hidden_size=32, num_heads=4)
    y = module(x)
    assert y.shape == x.shape


def test_attention_passes_bs3hd_to_dispatch(monkeypatch) -> None:
    requires_torch()

    import torch

    from novadit.modules import NovaDiTAttention, dit_attention

    observed_shape: tuple[int, ...] | None = None

    def fake_attention(qkv: torch.Tensor, **kwargs) -> torch.Tensor:
        nonlocal observed_shape
        observed_shape = tuple(qkv.shape)
        return qkv[:, :, 0]

    monkeypatch.setattr(
        dit_attention,
        "run_attention_qkvpacked",
        fake_attention,
    )

    x = torch.randn(2, 8, 32)
    module = NovaDiTAttention(hidden_size=32, num_heads=4)
    y = module(x)

    assert observed_shape == (2, 8, 3, 4, 8)
    assert y.shape == x.shape
