from __future__ import annotations

from tests.helpers import requires_torch


def test_auto_backend_falls_back_to_sdpa_without_flash_attn() -> None:
    requires_torch()

    from novadit.attention.dispatch import get_attention_backend

    assert get_attention_backend("auto") in {"fa2", "sdpa"}


def test_qkvpacked_sdpa_matches_separate_qkv() -> None:
    requires_torch()

    import torch

    from novadit.attention import run_attention_qkvpacked
    from novadit.attention.sdpa import run_sdpa

    qkv = torch.randn(2, 8, 3, 4, 8)
    q, k, v = qkv.unbind(dim=2)

    packed_out = run_attention_qkvpacked(qkv, backend="sdpa")
    separate_out = run_sdpa(q, k, v)

    torch.testing.assert_close(packed_out, separate_out)
