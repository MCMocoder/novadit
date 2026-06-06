#pragma once

#ifdef NOVADIT_HAS_CUTLASS
#include <cuda.h>
#include <cudaTypedefs.h>

#if CUDA_VERSION >= 13000
typedef PFN_cuTensorMapEncodeTiled_v12000 PFN_cuTensorMapEncodeTiled;
typedef PFN_cuTensorMapEncodeIm2col_v12000 PFN_cuTensorMapEncodeIm2col;
#endif

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#endif
