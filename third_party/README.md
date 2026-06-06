# Third-Party Dependencies

该目录用于存放 vendored 外部依赖。

当前已接入：

- `cutlass/`

NovaDiT 当前通过 `third_party/cutlass/include` 同时提供：

- CUTLASS 头文件
- CuTe 头文件

默认版本由 `scripts/bootstrap_cutlass.sh` 管理，当前 tag 为 `v3.5.1`。
