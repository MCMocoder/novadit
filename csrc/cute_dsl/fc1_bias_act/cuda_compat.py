"""CUDA Python compatibility helpers for CuTe DSL entry points."""

from __future__ import annotations

import importlib
from types import ModuleType


def _load_cuda_with_stream() -> ModuleType:
    """Return a CUDA module that exposes the CUstream Python wrapper."""
    for module_name in (
        "cutlass.cute.cuda",
        "cuda.bindings.driver",
        "cuda.cuda",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "CUstream"):
            return module

    try:
        import cutlass
    except ImportError:
        cutlass = None

    cutlass_cuda = getattr(cutlass, "cuda", None)
    if cutlass_cuda is not None and hasattr(cutlass_cuda, "CUstream"):
        return cutlass_cuda

    raise ImportError(
        "Unable to locate a CUDA Python module exposing CUstream. Install a "
        "cuda-python package compatible with this CUTLASS CuTe DSL build, or "
        "use an environment where cuda.bindings.driver.CUstream is available."
    )


cuda = _load_cuda_with_stream()

CUstream = cuda.CUstream

__all__ = ["CUstream", "cuda"]
