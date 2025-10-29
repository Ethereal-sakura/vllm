# 增量编译工作流程

当你在 vLLM 的 C++/CUDA 内核（位于 `csrc/` 目录）进行开发时，每次修改后都用 `uv pip install -e .` 重新编译整个项目会非常耗时。借助 CMake 的增量编译工作流，只需在初始设置后编译必要的部分，大大加快迭代速度。本指南将介绍如何配置和使用这种工作流程，以配合你的 Python 可编辑安装。

## 前置条件

在开始设置增量构建之前：

1. **vLLM 可编辑安装：** 请确保你已经从源码以可编辑模式安装了 vLLM。初始安装时使用预编译的 wheel 可以加快速度，后续的内核编译会由 CMake 处理。

    ```console
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
    ```

2. **CUDA 工具包：** 确认 NVIDIA CUDA Toolkit 已正确安装，并且 `nvcc` 命令可以在你的 `PATH` 中找到。CMake 需要通过 `nvcc` 编译 CUDA 代码。你通常可以在 `$CUDA_HOME/bin/nvcc` 找到它，或者用 `which nvcc` 进行查询。如果遇到问题，请参考 [CUDA 官方安装指南](https://developer.nvidia.com/cuda-toolkit-archive) 和 vLLM 的 [GPU 安装文档](../getting_started/installation/gpu.md#troubleshooting) 获取帮助。同时，`CMAKE_CUDA_COMPILER` 变量需要指向你的 `nvcc` 路径（可在 `CMakeUserPresets.json` 设置）。

3. **编译工具：** 强烈建议安装 `ccache`，它能缓存编译结果，显著加快构建速度（例如：`sudo apt install ccache` 或 `conda install ccache`）。还要确保基本构建依赖项如 `cmake` 和 `ninja` 已安装。这些依赖可以通过 `requirements/build.txt` 或系统包管理器安装。

    ```console
    uv pip install -r requirements/build.txt --torch-backend=auto
    ```

## 配置 CMake 构建环境

增量编译流程由 CMake 管理。你可以通过在 vLLM 仓库根目录创建 `CMakeUserPresets.json` 文件来配置你的构建参数。

### 用辅助脚本生成 `CMakeUserPresets.json`

为简化配置，vLLM 提供了一个辅助脚本，可自动检测你的系统环境（如 CUDA 路径、Python 环境、CPU 核心数），并生成 `CMakeUserPresets.json`。

**运行脚本：**

切换到 vLLM 仓库根目录，执行下列命令：

```console
python tools/generate_cmake_presets.py
```

如果脚本无法自动确定某些路径（比如 `nvcc` 或你的 vLLM 开发环境下的 Python 可执行文件），它会提示你输入相关信息。按照屏幕提示操作即可。如果检测到已有 `CMakeUserPresets.json`，脚本会询问是否覆盖。

**强制覆盖已有文件：**

如果你不希望有交互式提示，可以加上 `--force-overwrite` 参数，自动覆盖旧文件：

```console
python tools/generate_cmake_presets.py --force-overwrite
```

这对于自动化脚本或 CI/CD 环境特别方便。

运行完毕后，根目录下会生成一个新的 `CMakeUserPresets.json` 文件。

### 示例 `CMakeUserPresets.json`

下面是一个可能生成的 `CMakeUserPresets.json` 示例，实际内容会根据你的系统环境和输入自动调整。

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 26,
        "patch": 1
    },
    "configurePresets": [
        {
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_PYTHON_EXECUTABLE": "/home/user/venvs/vllm/bin/python",
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "4",
                "CMAKE_JOB_POOLS": "compile=32"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 32
        }
    ]
}
```

**各配置项含义说明：**

- `CMAKE_CUDA_COMPILER`：你的 `nvcc` 路径，脚本会自动检测设置。
- `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER`：指定为 `ccache`（或 `sccache`），能大幅加速编译。请确保已安装 `ccache`（如 `sudo apt install ccache` 或 `conda install ccache`）。脚本会默认设置。
- `VLLM_PYTHON_EXECUTABLE`：你的 vLLM 开发环境下 Python 的路径。脚本会根据环境自动设置或提示输入。
- `CMAKE_INSTALL_PREFIX: "${sourceDir}"`：指定编译好的组件安装到 vLLM 源码目录。这对可编辑安装非常重要，确保新编译的内核能被 Python 环境及时加载。
- `CMAKE_JOB_POOLS` 和 build presets 里的 `jobs`：控制构建并行度。脚本会根据你的 CPU 核心数自动设置。
- `binaryDir`：指定构建输出目录，比如 `cmake-build-release`。

## 用 CMake 编译和安装

配置好 `CMakeUserPresets.json` 后：

1. **初始化 CMake 构建环境：**
   根据你选择的 preset（如 `release`），配置构建系统，并在 `binaryDir` 目录下生成相关文件。

    ```console
    cmake --preset release
    ```

2. **编译并安装 vLLM 组件：**
   该命令会编译源码，并将生成的二进制文件安装到 vLLM 源码目录，使你的 Python 可编辑安装立即可用。

    ```console
    cmake --build --preset release --target install
    ```

3. **修改和重复编译：**
    现在你可以在 vLLM 可编辑安装环境下进行开发和测试了。每当你有新的更改，只需再次运行 CMake 构建命令，会只编译受影响的文件，节省大量时间。

    ```console
    cmake --build --preset release --target install
    ```

## 验证构建结果

编译成功后，你会在构建目录（比如 `cmake-build-release/`，如果用的是 `release` preset 和示例配置）看到生成的文件：

```console
> ls cmake-build-release/
bin             cmake_install.cmake      _deps                                machete_generation.log
build.ninja     CPackConfig.cmake        detect_cuda_compute_capabilities.cu  marlin_generation.log
_C.abi3.so      CPackSourceConfig.cmake  detect_cuda_version.cc               _moe_C.abi3.so
CMakeCache.txt  ctest                    _flashmla_C.abi3.so                  moe_marlin_generation.log
CMakeFiles      cumem_allocator.abi3.so  install_local_manifest.txt           vllm-flash-attn
```

通过 `cmake --build ... --target install` 命令，编译好的共享库（如 `_C.abi3.so`, `_moe_C.abi3.so` 等）会被复制到源码树里的 vllm 包目录，更新了你的可编辑安装环境，使新编译的内核立刻可用。

## 其他实用建议

- **调整并行度：** 根据你的内存和 CPU 情况，适当调整 `CMAKE_JOB_POOLS` 和 `jobs` 数量。并行度过高可能导致系统负载过重或内存不足，反而拖慢编译；设置太低又无法充分利用硬件资源。
- **必要时清理构建目录：** 如果遇到持续或异常的构建错误，尤其是在大幅更新或切换分支后，建议删除 CMake 构建目录（比如 `rm -rf cmake-build-release`），然后重新运行 `cmake --preset` 和 `cmake --build`。
- **指定目标模块编译：** 如果只需快速迭代某个模块，可以只编译相关 target，而不是全部 `install`。不过 `install` 能确保所有必要组件都被更新。更多高级 target 管理请参考 CMake 官方文档。