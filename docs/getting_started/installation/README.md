# 安装指南

vLLM 支持以下硬件平台：

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md#nvidia-cuda)
    - [AMD ROCm](gpu.md#amd-rocm)
    - [Intel XPU](gpu.md#intel-xpu)
- [CPU](cpu.md)
    - [Intel/AMD x86 架构](cpu.md#intelamd-x86)
    - [ARM AArch64 架构](cpu.md#arm-aarch64)
    - [Apple 芯片](cpu.md#apple-silicon)
    - [IBM Z (S390X)](cpu.md#ibm-z-s390x)
- [Google TPU](google_tpu.md)

## 硬件插件

以下后端插件**不在**主仓库 `vllm` 内，遵循
[Hardware-Pluggable RFC](../../design/plugin_system.md) 设计。

| 加速器 | PyPI / 包名 | 仓库地址 |
|-------------|----------------|------------|
| Ascend NPU | `vllm-ascend` | <https://github.com/vllm-project/vllm-ascend> |
| Intel Gaudi (HPU) | 无 PyPI 包，请源码安装 | <https://github.com/vllm-project/vllm-gaudi> |
| MetaX MACA GPU | 无 PyPI 包，请源码安装 | <https://github.com/MetaX-MACA/vLLM-metax> |
| Rebellions ATOM / REBEL NPU | `vllm-rbln` | <https://github.com/rebellions-sw/vllm-rbln> |
| IBM Spyre AIU | `vllm-spyre` | <https://github.com/vllm-project/vllm-spyre> |
| Cambricon MLU | `vllm-mlu` | <https://github.com/Cambricon/vllm-mlu> |