from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
CUTLASS_ROOT = Path(
    os.environ.get("NOVADIT_CUTLASS_PATH", ROOT / "third_party" / "cutlass")
).resolve()
OFFLINE_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0;10.0;10.3;11.0;12.0;12.1+PTX"
DEBUG_BUILD = os.environ.get("NOVADIT_DEBUG", "0") == "1"


def has_cutlass() -> bool:
    return (CUTLASS_ROOT / "include" / "cutlass").is_dir() and (
        CUTLASS_ROOT / "include" / "cute"
    ).is_dir()


def get_extensions_and_cmdclass():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except Exception:
        return [], {}

    if CUDA_HOME is None and os.environ.get("NOVADIT_FORCE_CUDA", "0") != "1":
        return [], {"build_ext": BuildExtension}

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = (
            "native" if torch.cuda.device_count() > 0 else OFFLINE_CUDA_ARCH_LIST
        )

    include_dirs = [str(ROOT / "csrc")]
    if has_cutlass():
        include_dirs.extend(
            [
                str(CUTLASS_ROOT / "include"),
                str(CUTLASS_ROOT / "tools" / "util" / "include"),
            ]
        )

    common_nvcc_args = [
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    if DEBUG_BUILD:
        cxx_args = ["-O0", "-g3", "-fno-omit-frame-pointer"]
        nvcc_args = ["-O0", "-g", "-G", *common_nvcc_args]
    else:
        cxx_args = ["-O3"]
        nvcc_args = ["-O3", "--use_fast_math", "-lineinfo", *common_nvcc_args]
    if has_cutlass():
        cxx_args.append("-DNOVADIT_HAS_CUTLASS=1")
        nvcc_args.extend(
            [
                "-DNOVADIT_HAS_CUTLASS=1",
                "-DCUTE_USE_PACKED_TUPLE=1",
            ]
        )

    sources = [
        "csrc/bindings.cpp",
        "csrc/ops/adaln_kernel.cu",
        "csrc/ops/qkv_pack_kernel.cu",
        "csrc/ops/residual_kernel.cu",
        "csrc/ops/gate_act_kernel.cu",
    ]

    extension = CUDAExtension(
        name="novadit_cuda",
        sources=[str(ROOT / source) for source in sources],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extensions_and_cmdclass()


setup(
    name="novadit",
    version="0.1.0",
    description="NovaDiT v0.1 scaffold for PyTorch DiT training with fused CUDA ops.",
    packages=find_packages(include=["novadit", "novadit.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
