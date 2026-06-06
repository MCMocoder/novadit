# CUTLASS Integration Notes

NovaDiT 当前将 CUTLASS 源码放在：

```text
third_party/cutlass
```

而不是直接塞进 `csrc/`。

这样做的目的：

- 保持 `csrc/` 只存放 NovaDiT 自己的 binding 和 kernels
- 让 CUTLASS / CuTe 以标准 include 目录方式接入
- 后续更容易升级 CUTLASS 版本

当前构建会自动加入以下 include 路径：

- `third_party/cutlass/include`
- `third_party/cutlass/tools/util/include`

因此 `.cu` 文件后续可以直接包含：

```cpp
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
```

如果需要重新初始化依赖，请运行：

```bash
./scripts/bootstrap_cutlass.sh
```
