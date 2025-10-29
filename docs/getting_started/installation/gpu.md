# GPU

vLLM 是一个 Python 库，支持以下几种 GPU 类型。请选择你的 GPU 类型，查看针对不同厂商的安装说明：

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:installation"

## 系统要求

- 操作系统：Linux
- Python 版本：3.10 -- 3.13

!!! note
    vLLM 当前不原生支持 Windows。如果你需要在 Windows 上运行 vLLM，可以使用 Windows Subsystem for Linux（WSL）并安装兼容的 Linux 发行版，或者尝试一些社区维护的分支，例如 [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:requirements"

## 使用 Python 环境进行安装

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:set-up-using-python"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:set-up-using-python"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:set-up-using-python"

### 预编译的 wheel 包

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-wheels"

### 从源码构建 wheel 包

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-wheel-from-source"

## 使用 Docker 部署

### 预构建镜像

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-images"

### 从源码构建镜像

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-image-from-source"

## 支持的功能特性

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:supported-features"
