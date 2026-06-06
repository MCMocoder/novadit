from __future__ import annotations

AUTO_BACKEND = "auto"
SUPPORTED_BACKENDS = {"auto", "fa2", "fa3", "fa4", "sdpa", "cudnn_sdpa", "math"}

__all__ = ["AUTO_BACKEND", "SUPPORTED_BACKENDS"]
