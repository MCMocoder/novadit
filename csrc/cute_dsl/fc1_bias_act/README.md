# Fused Gated FC1 + Activation (CuTe DSL, SM120)

This folder owns the CuTe DSL implementation of the fused gated FC1 +
activation kernel tracked by the `fc1_bias_act` name.

## Scope

Current scope is **FC1 + gated activation only** (no FC2, no bias):

```text
gate_part  = x @ w_gate^T     # [M, I]
value_part = x @ w_value^T    # [M, I]
y          = value_part * act(gate_part)   # [M, I], BF16
```

`act` is `silu` or `gelu` (tanh approximation), selected at compile time by
`act_id` (a `cutlass.Constexpr`). The interleaved gated FC1 weight is split
into two distinct `[I, H]` row-major pointers (`w_gate`, `w_value`) before
being passed to the kernel; there are no bias pointers and no FC2 operands.
The FC2 hooks in the surrounding Python/C++ ABI remain unused for now.

## Kernel

- Target: SM120 GeForce, warp-level `mma.sync.aligned.m16n8k16` (BF16 in,
  FP32 acc).
- Mainloop: one shared A tile (`sA`) plus two B tiles per stage
  (`sB_gate`, `sB_value`), with **two** `cute.gemm` calls per K-block
  accumulating into `acc_value` and `acc_gate`. Mirrors the
  `PipelineTmaAsync` producer/consumer + `PipelineTmaStore` epilogue pattern
  from `dense_gemm.py`.
- Epilogue: gated activation computed on the FP32 accumulators in registers,
  cast to BF16, then r2s (`StMatrix`) + TMA store to GMEM.
- Entry: `FC1GatedActSM120BF16Kernel.__call__` (host/JIT) launches the
  `@cute.kernel kernel` device method.

## Layout

- `fc1_bias_act.py`: CuTe DSL source (`FC1GatedActSM120BF16Kernel`) and ABI
  constants.
- `export_aot.py`: AOT entry. Builds fake CuTe tensors via
  `make_fake_compact_tensor`, calls `cute.compile(...)`, then `export_to_c`.
- `aot_bridge.h`: C++ bridge contract (`ForwardParams` with the two-weight
  ABI) used by the PyTorch extension once AOT artifacts exist.
- `generated/`: destination for `fc1_bias_act_aot.h` and
  `fc1_bias_act_aot.o`.

The current C++ extension entry point validates inputs and raises an explicit
"generated launcher not wired yet" error; wiring the AOT launcher is a
separate task.

