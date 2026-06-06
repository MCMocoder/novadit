from __future__ import annotations

import platform


def get_cuda_arch_label() -> str | None:
    """Best-effort architecture detection for future backend dispatch."""
    machine = platform.machine().lower()
    if "arm" in machine:
        return "unknown"
    return None
