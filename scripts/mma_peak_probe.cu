// Pure mma.sync bf16 throughput probe for SM120 (no smem / no TMA / no epilogue).
//
// Sweeps independent accumulator chains (ILP) x warps per CTA to separate the
// tensor pipe's true ceiling from the per-warp dependency limit, expressed in
// ncu's units (sm__pipe_tensor_cycles_active pct_of_peak_sustained_active,
// whose peak = 4 pipes x 16-cycle occupancy = 0.25 HMMA/SM/cycle).
//
// Build: nvcc -arch=sm_120a -O3 -o mma_peak_probe scripts/mma_peak_probe.cu
// Run:   ./mma_peak_probe

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__device__ __forceinline__ void mma_bf16_m16n8k16(float d[4], const unsigned a[4],
                                                  const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int CHAIN>
__global__ void mma_peak_kernel(int iters, float *sink) {
  unsigned a[4] = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3};
  unsigned b[CHAIN][2];
  float acc[CHAIN][4];

#pragma unroll
  for (int i = 0; i < CHAIN; ++i) {
    b[i][0] = threadIdx.x + i;
    b[i][1] = threadIdx.x ^ (i + 1);
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  }

  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < CHAIN; ++i) mma_bf16_m16n8k16(acc[i], a, b[i]);
  }

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < CHAIN; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) s += acc[i][j];
  if (s == 12345.678f) sink[0] = s;
}

template <int CHAIN>
double run_case(int sms, int threads, int iters, int reps, float *sink,
                float *out_ms) {
  for (int r = 0; r < 2; ++r) mma_peak_kernel<CHAIN><<<sms, threads>>>(iters / 10, sink);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  float best = 1e9f;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(t0));
    mma_peak_kernel<CHAIN><<<sms, threads>>>(iters, sink);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms < best) best = ms;
  }
  *out_ms = best;
  const double warps = (double)(threads / 32);
  return (double)sms * warps * (double)iters * CHAIN;  // total HMMA
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  const double clk = prop.clockRate * 1e3;  // Hz
  float *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));

  printf("device: %s (%d SMs, %.0f MHz)\n", prop.name, sms, clk / 1e6);
  printf("ncu peak model: 0.250 HMMA/SM/cycle (4 pipes x 16-cycle occupancy)\n\n");
  printf("%6s %7s %10s %9s %11s %12s %9s\n", "warps", "chains", "ms", "G HMMA/s",
         "TFLOPS", "HMMA/SM/cyc", "% of peak");

  const int iters = 200000;
  struct { int warps, chains; } cases[] = {
      {16, 1}, {16, 2}, {16, 4}, {16, 8},
      {32, 4}, {32, 8},
      {8, 8},
  };

  for (auto c : cases) {
    const int threads = c.warps * 32;
    float ms = 0.f;
    double hmma = 0;
    switch (c.chains) {
      case 1: hmma = run_case<1>(sms, threads, iters, 5, sink, &ms); break;
      case 2: hmma = run_case<2>(sms, threads, iters, 5, sink, &ms); break;
      case 4: hmma = run_case<4>(sms, threads, iters, 5, sink, &ms); break;
      case 8: hmma = run_case<8>(sms, threads, iters, 5, sink, &ms); break;
      default: continue;
    }
    const double rate = hmma / (ms * 1e-3);
    const double per_sm_cyc = rate / sms / clk;
    printf("%6d %7d %10.3f %9.2f %11.1f %12.4f %8.1f%%\n", c.warps, c.chains, ms,
           rate / 1e9, hmma * 4096.0 / (ms * 1e-3) / 1e12, per_sm_cyc,
           100.0 * per_sm_cyc / 0.25);
  }

  cudaFree(sink);
  return 0;
}
