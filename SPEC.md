# NovaDiT v0.1 Spec：PyTorch Training + FlashAttention + Partial Fused Kernels

## 0. v0.1 总目标

实现一个面向 **Diffusion Transformer / DiT** 的 PyTorch 训练算子库 **NovaDiT v0.1**。

v0.1 必须支持：

```text
1. PyTorch training forward/backward
2. FlashAttention backend dispatch
3. Ampere / Ada / Hopper / Blackwell 架构识别
4. PyTorch SDPA fallback
5. 部分高价值 fused CUDA kernels
6. BF16 / FP16 training
7. correctness tests
8. benchmark
```

v0.1 的 fused kernel 目标不是“完整 fused DiT block”，而是先融合 **DiT 中最频繁、最 memory-bound、最容易从 kernel launch / HBM traffic 中获益的部分**。

---

# 1. v0.1 必须实现的 fused kernel

v0.1 选择以下 4 个 fused kernel。

## Kernel 1：Fused AdaLN / Modulated LayerNorm

### 名称

```text
novadit.ops.fused_adaln
```

### 优先级

最高。

### 原因

DiT 中 AdaLN / AdaLN-Zero 非常常见，通常形式类似：

```text
x_norm = LayerNorm(x)
x_mod  = x_norm * (1 + scale) + shift
```

这类操作包含：

```text
mean
variance
rsqrt
normalize
scale
shift
optional gate
```

如果用 PyTorch eager，会产生多个小 kernel 和多次 HBM 读写。它是 DiT-specific fused kernel 中最值得先做的部分。

### v0.1 支持形式

输入：

```python
x:     [B, S, H]
scale: [B, H] or [B, 1, H]
shift: [B, H] or [B, 1, H]
```

输出：

```python
y: [B, S, H]
```

计算：

```text
mean = mean(x, dim=-1)
var  = var(x, dim=-1)
xhat = (x - mean) * rsqrt(var + eps)
y    = xhat * (1 + scale) + shift
```

### API

```python
def fused_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    ...
```

### 支持 dtype

```text
输入:
- torch.float16
- torch.bfloat16

内部累加:
- float32

输出:
- 与 x.dtype 相同
```

### Backward

v0.1 必须支持 backward。

实现方式允许两阶段：

```text
P0:
    forward 使用自定义 CUDA kernel
    backward 使用 PyTorch reference fallback

P1:
    forward + backward 都使用自定义 CUDA kernel
```

但是最终 v0.1 Definition of Done 要求：

```text
fused_adaln 可以在 PyTorch training 中正常 backward。
```

如果时间不足，允许 backward 在 custom autograd 里调用 PyTorch fallback，但必须在 README 中标注：

```text
fused_adaln backward currently uses PyTorch fallback in v0.1.
```

---

## Kernel 2：Fused QKV Linear + Bias + QKV Pack

### 名称

```text
novadit.ops.fused_qkv_pack
```

### 优先级

最高。

### 原因

FlashAttention 需要的 Q/K/V layout 通常是：

```text
[B, S, num_heads, head_dim]
```

但普通 PyTorch Linear 输出通常是：

```text
[B, S, 3 * H]
```

然后还要做：

```text
bias add
view
reshape
permute
contiguous
split q/k/v
```

这些操作本身虽然不是 GEMM，但会造成额外 kernel launch、layout transform 和 HBM traffic。

v0.1 不要求你从零实现完整高性能 GEMM，但是必须至少实现一个 fused QKV pack kernel，把：

```text
qkv_linear_output + bias + reshape + split
```

融合成适合 FlashAttention 的 layout。

### v0.1 推荐实现方式

分两个版本：

## 版本 A：实际必须实现

先用 PyTorch / CUTLASS linear 得到 packed qkv：

```python
qkv = torch.nn.functional.linear(x, w_qkv, bias_qkv)
```

然后自定义 CUDA kernel 执行：

```text
[B, S, 3 * H] -> q, k, v
```

输出：

```text
q: [B, S, num_heads, head_dim]
k: [B, S, num_heads, head_dim]
v: [B, S, num_heads, head_dim]
```

这个 kernel 命名为：

```text
fused_qkv_pack
```

### API

```python
def fused_qkv_pack(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

### 计算逻辑

输入：

```text
qkv shape = [B, S, 3 * num_heads * head_dim]
```

输出：

```text
q[b, s, h, d] = qkv[b, s, 0 * H + h * D + d]
k[b, s, h, d] = qkv[b, s, 1 * H + h * D + d]
v[b, s, h, d] = qkv[b, s, 2 * H + h * D + d]
```

其中：

```text
H = num_heads * head_dim
D = head_dim
```

### 版本 B：v0.1 optional / v0.2 默认

后续替换为真正的 CUTLASS fused GEMM：

```text
x @ W_qkv + bias + QKV pack
```

接口预留：

```python
def fused_qkv_linear_pack(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

v0.1 中这个函数可以先 fallback 到：

```python
qkv = F.linear(x, weight, bias)
q, k, v = fused_qkv_pack(qkv, num_heads, head_dim)
```

但目录和 API 必须预留。

---

## Kernel 3：Fused Attention Output + Bias + Residual

### 名称

```text
novadit.ops.fused_out_residual
```

### 优先级

高。

### 原因

Attention 后通常有：

```text
attn_out
-> output projection
-> bias
-> dropout
-> residual add
```

v0.1 不要求 fused output projection GEMM，但必须融合 projection 后的 elementwise 部分：

```text
y = residual + bias + projected
```

如果训练中启用 dropout，则可以先 fallback 到 PyTorch dropout；v0.1 fused kernel 默认不处理 dropout mask。

### v0.1 支持形式

输入：

```python
x:        [B, S, H]  # projected attention output
residual: [B, S, H]
bias:     [H] or None
```

输出：

```python
y: [B, S, H]
```

计算：

```text
if bias is not None:
    y = residual + x + bias
else:
    y = residual + x
```

### API

```python
def fused_out_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

### Dropout 策略

v0.1 中：

```text
dropout_p == 0:
    使用 fused_out_residual

dropout_p > 0:
    使用 PyTorch fallback:
        y = residual + dropout(x + bias)
```

后续 v0.2 再实现：

```text
fused bias + dropout + residual
```

不要在 v0.1 中实现随机 dropout mask kernel，避免测试和 deterministic backward 复杂化。

---

## Kernel 4：Fused MLP Gate Activation

### 名称

```text
novadit.ops.fused_gate_act
```

### 优先级

中高。

### 原因

DiT block 中 MLP 常见形式：

```text
x -> Linear up/gate -> activation -> multiply gate -> Linear down
```

例如：

```text
SwiGLU:
    y = silu(x_gate) * x_up

GEGLU:
    y = gelu(x_gate) * x_up
```

如果用 PyTorch，会产生：

```text
chunk
activation
mul
```

多个 kernel。这个融合实现难度低，但收益稳定。

### v0.1 支持形式

输入：

```python
x: [B, S, 2 * I]
```

输出：

```python
y: [B, S, I]
```

计算：

```text
x_up, x_gate = split(x, dim=-1)

SwiGLU:
    y = x_up * silu(x_gate)

GEGLU:
    y = x_up * gelu(x_gate)
```

### API

```python
def fused_gate_act(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    ...
```

支持：

```text
activation = "silu"
activation = "gelu"
```

v0.1 默认：

```text
activation = "silu"
```

### Backward

v0.1 必须支持 PyTorch training。

允许：

```text
forward 使用 CUDA fused kernel
backward 使用 PyTorch fallback
```

更好：

```text
forward + backward 都自定义 CUDA kernel
```

但不是 v0.1 强制项。

---

# 2. v0.1 不做哪些 fused kernel

以下 kernel 不在 v0.1 实现：

```text
不做：
- 完整 fused FlashAttention
- 完整 attention block monolithic forward/backward
- fused QKV GEMM with CUTLASS mandatory implementation
- fused output projection GEMM mandatory implementation
- fused dropout random mask kernel
- fused MLP full GEMM
- fused optimizer
- FP8 GEMM
- FP4 GEMM
- Tensor Parallel fused communication kernel
```

原因：

```text
v0.1 的目标是训练可用、结构清晰、benchmark 可验证。
不要把范围扩展成完整 TransformerEngine。
```

---

# 3. 修改后的 v0.1 架构

目录结构改为：

```text
NovaDiT/
├── pyproject.toml
├── setup.py
├── README.md
├── novadit/
│   ├── __init__.py
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── dispatch.py
│   │   ├── backends.py
│   │   ├── sdpa.py
│   │   ├── flash_attn_compat.py
│   │   └── utils.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── dit_attention.py
│   │   ├── dit_mlp.py
│   │   └── dit_block.py
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── adaln.py
│   │   ├── qkv_pack.py
│   │   ├── residual.py
│   │   └── gate_act.py
│   ├── precision/
│   │   ├── __init__.py
│   │   └── policy.py
│   └── testing/
│       ├── __init__.py
│       └── reference.py
├── csrc/
│   ├── bindings.cpp
│   ├── common/
│   │   ├── cuda_check.h
│   │   ├── dispatch.h
│   │   └── type_dispatch.h
│   ├── ops/
│   │   ├── adaln_kernel.cu
│   │   ├── qkv_pack_kernel.cu
│   │   ├── residual_kernel.cu
│   │   └── gate_act_kernel.cu
│   └── cutlass/
│       └── README.md
├── tests/
│   ├── test_backend_select.py
│   ├── test_attention_forward.py
│   ├── test_attention_backward.py
│   ├── test_fused_adaln.py
│   ├── test_fused_qkv_pack.py
│   ├── test_fused_out_residual.py
│   ├── test_fused_gate_act.py
│   ├── test_sdpa_fallback.py
│   └── test_dtype.py
├── benchmarks/
│   ├── bench_attention.py
│   ├── bench_fused_adaln.py
│   ├── bench_fused_qkv_pack.py
│   ├── bench_fused_out_residual.py
│   └── bench_fused_gate_act.py
└── examples/
    ├── minimal_train.py
    ├── minimal_dit_attention.py
    └── minimal_dit_block.py
```

---

# 4. PyTorch Extension 规格

## 4.1 编译方式

使用 PyTorch C++/CUDA extension。

`setup.py` 中应包含：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="novadit",
    packages=["novadit"],
    ext_modules=[
        CUDAExtension(
            name="novadit_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ops/adaln_kernel.cu",
                "csrc/ops/qkv_pack_kernel.cu",
                "csrc/ops/residual_kernel.cu",
                "csrc/ops/gate_act_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

## 4.2 Extension import 规则

`novadit/ops/__init__.py`：

```python
try:
    import novadit_cuda
    HAS_NOVADIT_CUDA = True
except Exception:
    novadit_cuda = None
    HAS_NOVADIT_CUDA = False
```

要求：

```text
import novadit 不能因为 CUDA extension 未编译而失败。
```

如果 extension 不存在，则所有 fused op 自动 fallback 到 PyTorch reference。

---

# 5. Fused op Python wrapper 规格

每个 fused op 都必须有：

```text
1. CUDA path
2. PyTorch reference fallback
3. correctness test
4. benchmark
```

## 5.1 `novadit/ops/adaln.py`

```python
import torch
import torch.nn.functional as F

try:
    import novadit_cuda
except Exception:
    novadit_cuda = None


def adaln_ref(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    xhat = (x.float() - mean) * torch.rsqrt(var + eps)

    while scale.dim() < x.dim():
        scale = scale.unsqueeze(1)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(1)

    y = xhat * (1.0 + scale.float()) + shift.float()
    return y.to(dtype=x.dtype)


def fused_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if novadit_cuda is not None and x.is_cuda:
        return novadit_cuda.fused_adaln_forward(x, scale, shift, eps)
    return adaln_ref(x, scale, shift, eps)
```

---

## 5.2 `novadit/ops/qkv_pack.py`

```python
def qkv_pack_ref(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, three_h = qkv.shape
    hidden = num_heads * head_dim
    assert three_h == 3 * hidden

    qkv = qkv.view(b, s, 3, num_heads, head_dim)
    q = qkv[:, :, 0].contiguous()
    k = qkv[:, :, 1].contiguous()
    v = qkv[:, :, 2].contiguous()
    return q, k, v


def fused_qkv_pack(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if novadit_cuda is not None and qkv.is_cuda:
        return novadit_cuda.fused_qkv_pack_forward(qkv, num_heads, head_dim)
    return qkv_pack_ref(qkv, num_heads, head_dim)
```

---

## 5.3 `novadit/ops/residual.py`

```python
def out_residual_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None:
        return x + residual + bias.view(1, 1, -1)
    return x + residual


def fused_out_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if novadit_cuda is not None and x.is_cuda:
        return novadit_cuda.fused_out_residual_forward(x, residual, bias)
    return out_residual_ref(x, residual, bias)
```

---

## 5.4 `novadit/ops/gate_act.py`

```python
def gate_act_ref(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    x_up, x_gate = x.chunk(2, dim=-1)

    if activation == "silu":
        return x_up * torch.nn.functional.silu(x_gate)

    if activation == "gelu":
        return x_up * torch.nn.functional.gelu(x_gate)

    raise ValueError(f"Unsupported activation: {activation}")


def fused_gate_act(
    x: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    if novadit_cuda is not None and x.is_cuda:
        act_id = 0 if activation == "silu" else 1
        return novadit_cuda.fused_gate_act_forward(x, act_id)
    return gate_act_ref(x, activation)
```

---

# 6. NovaDiTAttention 修改后执行流程

`novadit/modules/dit_attention.py`

## 6.1 初始化

```python
class NovaDiTAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        qkv_bias: bool = True,
        out_bias: bool = True,
        dropout_p: float = 0.0,
        causal: bool = False,
        attention_backend: str = "auto",
        use_fused_qkv_pack: bool = True,
        use_fused_out_residual: bool = True,
        device=None,
        dtype=None,
    ):
        ...
```

## 6.2 forward

```python
def forward(
    self,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = x

    qkv = self.qkv_proj(x)

    if self.use_fused_qkv_pack:
        q, k, v = fused_qkv_pack(qkv, self.num_heads, self.head_dim)
    else:
        q, k, v = qkv_pack_ref(qkv, self.num_heads, self.head_dim)

    attn_out = run_attention(
        q,
        k,
        v,
        dropout_p=self.dropout_p if self.training else 0.0,
        causal=self.causal,
        attention_mask=attention_mask,
        backend=self.attention_backend,
    )

    attn_out = attn_out.reshape(x.shape[0], x.shape[1], self.hidden_size)

    projected = self.out_proj(attn_out)

    if self.dropout_p == 0.0 and self.use_fused_out_residual:
        out = fused_out_residual(projected, residual, None)
    else:
        out = residual + self.dropout(projected)

    return out
```

---

# 7. NovaDiTMLP v0.1

新增：

```text
novadit/modules/dit_mlp.py
```

## 7.1 类签名

```python
class NovaDiTMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = True,
        dropout_p: float = 0.0,
        use_fused_gate_act: bool = True,
        device=None,
        dtype=None,
    ):
        ...
```

## 7.2 forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x

    x = self.fc1(x)  # [B, S, 2 * intermediate_size]

    if self.use_fused_gate_act:
        x = fused_gate_act(x, activation=self.activation)
    else:
        x = gate_act_ref(x, activation=self.activation)

    x = self.fc2(x)

    if self.dropout_p > 0:
        x = self.dropout(x)

    return residual + x
```

---

# 8. NovaDiTBlock v0.1

新增：

```text
novadit/modules/dit_block.py
```

## 8.1 目标

提供一个最小 DiT block：

```text
x
-> fused AdaLN
-> NovaDiTAttention
-> fused AdaLN
-> NovaDiTMLP
```

## 8.2 类签名

```python
class NovaDiTBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.0,
        attention_backend: str = "auto",
        use_fused_adaln: bool = True,
        use_fused_qkv_pack: bool = True,
        use_fused_out_residual: bool = True,
        use_fused_gate_act: bool = True,
        device=None,
        dtype=None,
    ):
        ...
```

## 8.3 condition 输入

v0.1 不实现完整 DiT timestep embedding module，只要求 forward 接收已经生成好的 scale/shift：

```python
def forward(
    self,
    x: torch.Tensor,
    attn_scale: torch.Tensor,
    attn_shift: torch.Tensor,
    mlp_scale: torch.Tensor,
    mlp_shift: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

执行：

```python
residual = x
x_mod = fused_adaln(x, attn_scale, attn_shift)
x = self.attn(x_mod, attention_mask=attention_mask)
x = residual + (x - x_mod)  # 或者直接让 attn 内部处理 residual，二选一，保持一致

residual = x
x_mod = fused_adaln(x, mlp_scale, mlp_shift)
x = self.mlp(x_mod)
x = residual + (x - x_mod)
return x
```

更推荐简单实现：

```python
x = x + self.attn(fused_adaln(x, attn_scale, attn_shift))
x = x + self.mlp(fused_adaln(x, mlp_scale, mlp_shift))
```

但这要求 `NovaDiTAttention` 和 `NovaDiTMLP` 不再内部做 residual。为了避免重复 residual，v0.1 需要明确：

```text
NovaDiTAttention:
    默认不做 residual，除非 add_residual=True

NovaDiTMLP:
    默认不做 residual，除非 add_residual=True

NovaDiTBlock:
    统一负责 residual。
```

因此请调整模块参数：

```python
add_residual: bool = False
```

---

# 9. CUDA kernel 实现要求

## 9.1 通用要求

所有 kernel 必须满足：

```text
- 输入必须 contiguous
- 不支持非 contiguous 时自动 .contiguous()
- 支持 fp16 / bf16
- 内部必要 reduction 使用 fp32
- 输出 dtype 与输入 x dtype 一致
- 对 shape 做 TORCH_CHECK
- 对 dtype 做 TORCH_CHECK
- 对 device 做 TORCH_CHECK
```

## 9.2 `fused_out_residual` kernel

这是最简单 kernel，必须优先实现。

每个 thread 处理一个元素：

```cpp
y[i] = x[i] + residual[i] + bias[h]
```

其中：

```cpp
h = i % hidden_size
```

## 9.3 `fused_qkv_pack` kernel

每个 thread 处理一个输出元素。

可以用 flattened index：

```cpp
idx -> b, s, h, d
```

然后：

```cpp
q_idx = ((b * S + s) * NH + h) * HD + d
qkv_q_idx = ((b * S + s) * (3 * H)) + h * HD + d
qkv_k_idx = ((b * S + s) * (3 * H)) + H + h * HD + d
qkv_v_idx = ((b * S + s) * (3 * H)) + 2 * H + h * HD + d
```

写入：

```cpp
q[q_idx] = qkv[qkv_q_idx]
k[q_idx] = qkv[qkv_k_idx]
v[q_idx] = qkv[qkv_v_idx]
```

## 9.4 `fused_gate_act` kernel

每个 thread 处理一个输出元素：

```cpp
up   = x[..., i]
gate = x[..., I + i]
```

SwiGLU：

```cpp
silu = gate / (1 + exp(-gate))
y = up * silu
```

GEGLU：

```cpp
gelu = 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))
y = up * gelu
```

v0.1 可以使用 approximate GELU。

## 9.5 `fused_adaln` kernel

这个 kernel 稍复杂。

推荐 v0.1 限制：

```text
hidden_size <= 8192
每个 block 处理一个 [B, S] row
blockDim = 256 or 512
```

每个 block 计算一行：

```text
row = x[b, s, :]
```

步骤：

```text
1. block reduce sum
2. mean = sum / H
3. block reduce squared diff
4. var = sq_sum / H
5. inv_std = rsqrt(var + eps)
6. y[h] = ((x[h] - mean) * inv_std) * (1 + scale[b, h]) + shift[b, h]
```

scale/shift 支持：

```text
[B, H]
[B, 1, H]
```

v0.1 可以在 Python wrapper 中统一 reshape 成：

```text
[B, H]
```

再传入 CUDA。

---

# 10. 测试要求

## 10.1 fused kernel correctness

每个 fused op 都要和 reference 对比。

### `test_fused_adaln.py`

测试 shape：

```text
B=2, S=128, H=768
B=1, S=1024, H=1152
B=4, S=256, H=2048
```

dtype：

```text
fp16
bf16
```

误差：

```text
fp16: rtol=1e-2, atol=1e-2
bf16: rtol=2e-2, atol=2e-2
```

### `test_fused_qkv_pack.py`

检查：

```text
q/k/v shape 正确
q/k/v 与 reference 完全相等或 allclose
q/k/v contiguous
```

### `test_fused_out_residual.py`

检查：

```text
无 bias
有 bias
fp16
bf16
```

### `test_fused_gate_act.py`

检查：

```text
SwiGLU
GEGLU
fp16
bf16
```

---

## 10.2 backward test

由于部分 fused kernel backward 可以 fallback 到 PyTorch reference，测试重点是：

```text
NovaDiTAttention backward 可用
NovaDiTMLP backward 可用
NovaDiTBlock backward 可用
```

测试：

```python
x.requires_grad_(True)
out = module(x)
loss = out.float().pow(2).mean()
loss.backward()
```

检查：

```text
x.grad is not None
所有 Linear 参数 grad is not None
无 NaN / Inf
```

对于 `fused_adaln`，如果用 custom autograd fallback backward，需要单独测：

```text
scale.grad is not None
shift.grad is not None
```

---

# 11. Benchmark 要求

每个 fused kernel 都要有 benchmark。

## 11.1 `bench_fused_adaln.py`

比较：

```text
PyTorch reference AdaLN
NovaDiT fused_adaln
```

shape：

```text
[B=2, S=1024, H=1152]
[B=4, S=4096, H=1536]
[B=1, S=16384, H=768]
```

指标：

```text
latency ms
speedup
effective bandwidth
max memory allocated
```

## 11.2 `bench_fused_qkv_pack.py`

比较：

```text
PyTorch view/split/contiguous
NovaDiT fused_qkv_pack
```

shape：

```text
B=2, S=1024, H=1152, heads=16
B=1, S=4096, H=2048, heads=16
```

## 11.3 `bench_fused_out_residual.py`

比较：

```text
PyTorch x + residual + bias
NovaDiT fused_out_residual
```

## 11.4 `bench_fused_gate_act.py`

比较：

```text
PyTorch chunk + silu/gelu + multiply
NovaDiT fused_gate_act
```

## 11.5 `bench_attention.py`

比较：

```text
NovaDiTAttention with fused_qkv_pack=False, fused_out_residual=False
NovaDiTAttention with fused_qkv_pack=True, fused_out_residual=True
```

输出：

```text
selected attention backend
latency
speedup
max memory allocated
```

---

# 12. 修改后的 Definition of Done

NovaDiT v0.1 完成标准：

```text
1. import novadit 成功
2. CUDA extension 可编译
3. CUDA extension 不存在时自动 fallback 到 PyTorch
4. fused_adaln forward 可用
5. fused_qkv_pack forward 可用
6. fused_out_residual forward 可用
7. fused_gate_act forward 可用
8. NovaDiTAttention forward/backward 可用
9. NovaDiTMLP forward/backward 可用
10. NovaDiTBlock forward/backward 可用
11. FlashAttention backend auto dispatch 可用
12. SDPA fallback 可用
13. pytest tests/ 通过
14. benchmark 脚本可运行
15. README 明确说明哪些 kernel 是 fused，哪些 backward 是 fallback
```

---

# 13. 给 Codex 的完整任务提示

可以直接复制下面这段给 Codex：

```text
请实现 NovaDiT v0.1。

目标：
实现一个 PyTorch training-compatible 的 DiT attention/block package，支持 FlashAttention backend dispatch，并实现 4 个高价值 fused CUDA kernels。

必须实现的 fused kernels：
1. fused_adaln
   - 输入 x [B,S,H], scale [B,H] or [B,1,H], shift [B,H] or [B,1,H]
   - 输出 y [B,S,H]
   - 计算 LayerNorm + modulation:
     y = LN(x) * (1 + scale) + shift
   - 支持 fp16/bf16
   - 内部 reduction 用 fp32
   - CUDA forward 必须实现
   - backward 可以 PyTorch fallback，但训练必须可用

2. fused_qkv_pack
   - 输入 qkv [B,S,3*H]
   - 输出 q,k,v each [B,S,num_heads,head_dim]
   - 支持 fp16/bf16
   - CUDA forward 必须实现
   - 用于 FlashAttention 前的 layout packing

3. fused_out_residual
   - 输入 x [B,S,H], residual [B,S,H], bias [H] or None
   - 输出 y = x + residual + bias
   - 支持 fp16/bf16
   - CUDA forward 必须实现

4. fused_gate_act
   - 输入 x [B,S,2*I]
   - 输出 y [B,S,I]
   - 支持 activation='silu' 和 activation='gelu'
   - 计算 SwiGLU/GEGLU:
     y = x_up * act(x_gate)
   - 支持 fp16/bf16
   - CUDA forward 必须实现
   - backward 可以 PyTorch fallback，但训练必须可用

项目结构：
- novadit/attention: FlashAttention dispatch + SDPA fallback
- novadit/ops: Python wrappers and PyTorch fallback
- novadit/modules: NovaDiTAttention, NovaDiTMLP, NovaDiTBlock
- csrc: C++/CUDA extension
- tests: correctness and backward tests
- benchmarks: microbenchmarks

FlashAttention backend：
- attention_backend 支持 auto/fa2/fa3/fa4/sdpa/cudnn_sdpa/math
- Ampere/Ada: fa2 -> sdpa
- Hopper: fa3 -> fa2 -> sdpa
- Blackwell: fa4 -> cudnn_sdpa -> sdpa
- 所有 FlashAttention import 必须 lazy import
- 未安装 flash-attn 时 import novadit 不能失败
- attention_mask 存在而 FA runner 不支持时，auto fallback 到 sdpa

NovaDiTAttention：
- 输入 x [B,S,H]
- qkv_proj 使用 torch.nn.Linear
- qkv output 使用 fused_qkv_pack 变为 q/k/v
- attention core 使用 run_attention dispatch
- out_proj 使用 torch.nn.Linear
- dropout_p == 0 时使用 fused_out_residual
- 默认 causal=False
- 支持 forward/backward

NovaDiTMLP：
- fc1 输出 [B,S,2*intermediate]
- 使用 fused_gate_act
- fc2 输出 [B,S,H]
- 支持 forward/backward

NovaDiTBlock：
- 使用 fused_adaln + NovaDiTAttention + fused_adaln + NovaDiTMLP
- forward 接收：
  x, attn_scale, attn_shift, mlp_scale, mlp_shift
- residual 由 NovaDiTBlock 统一管理
- Attention/MLP 子模块默认不内部加 residual，避免重复 residual

要求：
- Python 3.10+
- PyTorch 2.3+
- CUDA extension 使用 torch.utils.cpp_extension
- 所有 CUDA op 都有 PyTorch fallback
- 所有 CUDA 测试无 CUDA 时 skip
- 所有 FlashAttention 测试未安装时 skip
- pytest tests/ 必须通过
- benchmark 脚本必须能运行
- README 必须说明 v0.1 fused kernel 范围和未实现内容

不要实现：
- 完整 FlashAttention CUDA kernel
- 完整 fused attention backward
- FP8/FP4 GEMM
- Megatron-Core tensor parallel
- NCCL overlap
- fused dropout mask kernel
- full CUTLASS fused GEMM

请优先实现：
1. 项目结构
2. setup.py + csrc/bindings.cpp
3. fused_out_residual CUDA kernel
4. fused_qkv_pack CUDA kernel
5. fused_gate_act CUDA kernel
6. fused_adaln CUDA kernel
7. Python fallback wrappers
8. NovaDiTAttention
9. NovaDiTMLP
10. NovaDiTBlock
11. Tests
12. Benchmarks
```

---

# 14. 为什么这 4 个 kernel 最适合 v0.1

这 4 个 kernel 的选择逻辑是：

| Kernel             | 价值 | 实现风险 | 是否 DiT-specific |
| ------------------ | -: | ---: | --------------: |
| fused_adaln        | 很高 |    中 |              很强 |
| fused_qkv_pack     |  高 |    低 |               中 |
| fused_out_residual | 中高 |    低 |              通用 |
| fused_gate_act     |  高 |    低 |              中高 |

不选完整 fused attention 的原因：

```text
FlashAttention 已经高度优化，v0.1 重写没有意义。
NovaDiT 更应该融合 FlashAttention 前后的 DiT-specific 部分。
```

不选完整 fused GEMM 的原因：

```text
CUTLASS fused GEMM 价值很高，但 v0.1 如果强制完成 QKV GEMM + backward，会显著拖慢项目进度。
先实现 QKV pack，再在 v0.2 替换为 CUTLASS fused QKV GEMM。
```

不选 fused dropout 的原因：

```text
dropout 涉及随机数、mask 保存、determinism、backward，v0.1 性价比低。
DiT 训练中很多配置 dropout_p=0，因此 v0.1 可以先跳过。
```

---

# 15. v0.1 的最终定位

修改后的 NovaDiT v0.1 应该定位为：

```text
一个 PyTorch 训练可用的 DiT block runtime，
包含架构感知 FlashAttention backend dispatch，
并实现 AdaLN、QKV layout packing、attention output residual、
MLP gate activation 四类高价值 fused CUDA kernels。
```

这比单纯包装 FlashAttention 更像真实 AI infra 项目，同时又不会把范围扩大到无法完成。
