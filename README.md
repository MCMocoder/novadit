# NovaDiT

NovaDiT 是一个面向 Diffusion Transformer / DiT 的 PyTorch 训练算子库工程骨架。

当前仓库已完成：

- 项目目录初始化
- Python 包与 CUDA extension 构建入口
- CUTLASS / CuTe 本地依赖接入
- PyTorch fallback 代码骨架
- VS Code 开发与调试配置
- tests / benchmarks / examples 脚手架

当前仓库尚未完成完整的 v0.1 CUDA kernel 实现；`csrc/` 中的导出函数已经预留，后续可以在不改动工程结构的前提下逐步补齐。

## 目录

```text
novadit/
  attention/
  modules/
  ops/
  precision/
  testing/
csrc/
third_party/
tests/
benchmarks/
examples/
.vscode/
```

## 开发环境

推荐 Python 3.10+，并先安装 PyTorch 2.3+。

```bash
uv venv .venv
uv sync --extra dev
```

项目现在默认使用 `uv` 管理 `.venv` 和 PyTorch 依赖。

- 虚拟环境位置：`.venv/`
- `uv` 缓存目录：`.uv-cache/`
- 依赖锁文件：`uv.lock`

常用命令：

```bash
uv run python -c "import torch; print(torch.__version__)"
uv run pytest tests -q
uv run python setup.py build_ext --inplace
```

AdaLN benchmark 支持分别测试 forward、forward + backward 全链路、
纯 backward、dx kernel 和 dparam kernel：

```bash
uv run python benchmarks/bench_fused_adaln.py --mode all
uv run python benchmarks/bench_fused_adaln.py --mode full_chain
uv run python benchmarks/bench_fused_adaln.py --mode backward
```

`full_chain` 每轮重新执行 forward 并计算 `dx/dscale/dshift`；纯 backward
只计时反向执行，forward graph 会提前构建。报告中的
`backward_dx` 单独比较 CUDA dx kernel 与 PyTorch 公式实现；
`backward_dparam` 单独比较 CUDA dscale/dshift kernel 与 PyTorch reduction。
误差报告包含 `max_abs`、`RMSE`、`max_rel` 和 relative L2，并对完整
backward 分别列出 `dx`、`dscale`、`dshift`。

AdaLN dparam kernel 使用 FP32 累加，并且独立
`fused_adaln_backward_dparam` API 始终返回 FP32 的 `dscale/dshift`。
进入 PyTorch autograd 后，框架会在归并到 fp16/bf16 叶子参数时自动转换为
参数 dtype；若优化器需要保留 FP32 梯度，应使用 FP32 modulation 参数。

DiT block/stack 全链路 benchmark 会执行 SDPA、QKV/输出投影、MLP、
两次 AdaLN，并在训练模式下计算输入、modulation 和所有模型参数梯度：

```bash
uv run python benchmarks/bench_dit_full_chain.py --mode all
uv run python benchmarks/bench_dit_full_chain.py \
  --batch 2 --seq 256 --hidden 768 --heads 12 --depth 4
```

当前训练可用的 fused 路径为 AdaLN forward/backward。QKV pack 和 gate-act
在补齐 backward kernel 前保持 PyTorch 实现，benchmark 报告会明确注明，
避免将尚未实现的融合收益计入 DiT 全链路结果。

`NovaDiTAttention` 的 QKV projection 会直接 reshape 为 FlashAttention 的
`[B, S, 3, num_heads, head_dim]`（BS3HD）布局。安装 FlashAttention 后，
无 mask 的 CUDA fp16/bf16 输入会调用 `flash_attn_qkvpacked_func`；其他情况
从同一 packed view 拆出 Q/K/V 并回退到 PyTorch SDPA。

如果本机已安装 CUDA Toolkit，并且希望构建扩展：

```bash
uv run python setup.py build_ext --inplace
```

## CUTLASS / CuTe

仓库已将 CUTLASS 以 vendored dependency 的方式接入：

```text
third_party/cutlass/
  include/cutlass/
  include/cute/
```

默认构建会自动从以下位置寻找头文件：

- `third_party/cutlass/include`
- `third_party/cutlass/tools/util/include`

如果你想改用外部 CUTLASS 路径，可以设置：

```bash
export NOVADIT_CUTLASS_PATH=/path/to/cutlass
```

当前 `setup.py` 已为 CUTLASS/CuTe 预置常用 NVCC 参数：

- `--expt-relaxed-constexpr`
- `--expt-extended-lambda`
- `-DCUTE_USE_PACKED_TUPLE=1`

因此后续 `.cu` 文件可以直接使用：

```cpp
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
```

如果需要重新拉取或更新 CUTLASS，可使用：

```bash
./scripts/bootstrap_cutlass.sh
```

如果暂时没有 CUDA 环境，包仍然可以通过 PyTorch fallback 路径工作，`import novadit` 不会因为 extension 缺失而失败。

## VS Code

仓库已经包含：

- `.vscode/settings.json`
- `.vscode/tasks.json`
- `.vscode/launch.json`
- `.vscode/extensions.json`

可直接用于：

- 运行当前 pytest 文件
- 调试 `examples/minimal_train.py`
- 原地构建 CUDA extension
- 让 C++ IntelliSense 识别 CUTLASS / CuTe 头文件

### PyTorch / CUDA 断点调试

先在 VS Code 中安装仓库推荐的 `CodeLLDB` 和
`NVIDIA Nsight Visual Studio Code Edition` 插件，然后从
**Run and Debug** 选择：

- `PyTorch Python: AdaLN Driver`：调试 Python/autograd 调用路径。
- `PyTorch Native C++: AdaLN (CodeLLDB)`：调试 PyTorch extension 的
  C++ host 函数，例如 `fused_adaln_backward_dx`。
- `CUDA Kernel: AdaLN (Nsight CUDA-GDB)`：调试 CUDA device kernel，
  例如 `fused_adaln_backward_dx_kernel`。

这些配置都会先执行 `Build CUDA extension debug inplace`。该任务通过
`NOVADIT_DEBUG=1` 使用 C++ `-O0 -g3` 和 NVCC `-O0 -g -G`
生成可断点调试的扩展。CUDA-GDB 默认在第一个 kernel launch 停下，继续执行
前可以在 `csrc/ops/adaln_kernel.cu` 的目标 kernel 内设置源码断点。

CodeLLDB 和 CUDA-GDB 是两个互补的原生调试会话，不要同时附加到同一个
Python 进程，否则两个 debugger 会争用进程控制。需要跨层定位时，先用
Python/CodeLLDB 找到 host launch 参数，再单独启动 CUDA-GDB 检查线程、
block 和 device 变量。

### Nsight 性能评估

从 **Tasks: Run Task** 运行：

- `Profile AdaLN - Nsight Systems`：生成
  `build/profiles/adaln_nsys.nsys-rep`，分析 CPU/CUDA/NVTX 时间线。
- `Report AdaLN - Nsight Systems Stats`：在终端输出 Systems 汇总。
- `Profile AdaLN Kernel - Nsight Compute`：生成
  `build/profiles/adaln_ncu.ncu-rep`，采集 AdaLN kernel 指标。

性能任务会先执行 release 构建，避免 `-G` 改变 kernel 性能。本机配置固定
使用 CUDA 12.9 自带的 Nsight Systems 2025.1 和 Nsight Compute 2025.2，
而不是 `PATH` 中可能存在的旧版 `ncu`。

如果 Nsight Compute 报 `ERR_NVGPUCTRPERM`，说明当前用户没有读取 NVIDIA
GPU performance counters 的权限，需要由系统管理员按 NVIDIA 的
`ERR_NVGPUCTRPERM` 指引启用后再运行；Nsight Systems 时间线不依赖该权限。

## clangd / CMake

如果你在 `csrc/ops/*.cu` 里看见 `torch/extension.h` 找不到，先让项目生成编译数据库：

```bash
cmake -S . -B build -DPython3_EXECUTABLE=/path/to/python-with-torch -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

然后让 `clangd` 读取：

```text
build/clangd/compile_commands.json
```

仓库已提供：

- [CMakeLists.txt](/home/steven/novadit/CMakeLists.txt)
- [.clangd](/home/steven/novadit/.clangd)
- VS Code task: `Configure C++ build`

注意：这里的 `Python3_EXECUTABLE` 必须指向已经安装 `torch` 的 Python 环境，否则 CMake 无法解析 PyTorch 头文件和 ABI 配置。

在 VS Code 里，仓库任务已经改成自动跟随“当前选中的 Python 解释器”，所以先选中装有 `torch` 的环境，再运行 `Configure C++ build` 即可。
现在默认解释器就是项目内的 `.venv/bin/python`，并由 `uv` 负责维护。
如果诊断还停留在旧的 `nvcc` 参数上，重载一下 `clangd` 或重开工作区即可。

## 下一步

下一阶段建议优先实现：

1. `fused_out_residual`
2. `fused_qkv_pack`
3. `fused_gate_act`
4. `fused_adaln`
5. 完整 correctness tests 与 benchmark
