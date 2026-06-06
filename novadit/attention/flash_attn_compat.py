from __future__ import annotations

from importlib.util import find_spec


def has_flash_attn() -> bool:
    return find_spec("flash_attn") is not None


def import_flash_attn():
    if not has_flash_attn():
        return None
    import flash_attn  # type: ignore

    return flash_attn
