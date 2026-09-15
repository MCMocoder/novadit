# NovaDiT CuTe DSL Kernels

This directory is reserved for kernels authored with the CUTLASS CuTe Python
DSL. Native PyTorch extension entry points remain under `csrc/ops`; each CuTe
DSL kernel owns a subdirectory here with:

- Python DSL source.
- An AOT export script.
- A `generated/` directory for exported headers and objects.
- Notes describing the C++ bridge expected by the PyTorch extension.

The generated artifacts are intentionally not checked in until a kernel is
implemented and exported for the target CUDA/CUTLASS toolchain.
