# --8<-- [start:installation]

vLLM 内置了预编译好的 C++ 和 CUDA (12.8) 二进制文件。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU：计算能力（compute capability）7.0及以上（如 V100、T4、RTX20xx、A100、L4、H100 等）

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

!!! note
    通过 `conda` 安装的 PyTorch 会静态链接 `NCCL` 库，这可能导致 vLLM 在使用 `NCCL` 时出现问题。详细信息请参考 <https://github.com/vllm-project/vllm/issues/8420>

为了获得最佳性能，vLLM 需要编译大量 cuda 内核。但这种编译过程会导致不同 CUDA 版本、PyTorch 版本之间出现二进制不兼容，即使是同一个 PyTorch 版本，不同的编译配置也可能不兼容。

因此，建议在一个**全新环境**下安装 vLLM。如果你的 CUDA 版本不同，或者需要使用现有的 PyTorch 安装，必须从源码编译 vLLM。具体步骤请见[下文](#build-wheel-from-source)。

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

```bash
uv pip install vllm --torch-backend=auto
```

??? console "pip"
    ```bash
    # 使用 CUDA 12.8 安装 vLLM
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
    ```

我们推荐使用 `uv`，它可以通过 `--torch-backend=auto`（或设置环境变量 `UV_TORCH_BACKEND=auto`），[自动检测已安装的 CUDA 驱动版本，在运行时选择匹配的 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)。如果需要指定后端（比如 `cu126`），可以设置 `--torch-backend=cu126`（或环境变量 `UV_TORCH_BACKEND=cu126`）。如果遇到问题，先运行 `uv self update` 更新 `uv` 再试。

!!! note
    NVIDIA Blackwell 系列 GPU（如 B200、GB200）至少需要 CUDA 12.8，因此安装 PyTorch 时请确保版本不低于 12.8。PyTorch 官方提供了[专用页面](https://pytorch.org/get-started/locally/)，可以根据你的目标配置查询合适的 pip 安装命令。

目前 vLLM 的二进制文件默认使用 CUDA 12.8 和 PyTorch 公共版本编译。我们也提供了用 CUDA 12.6、11.8 以及相应 PyTorch 公共版本编译的 vLLM 二进制：

```bash
# 安装指定 CUDA 版本（如 11.8 或 12.6）的 vLLM
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=118 # 或 126
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

#### 安装最新代码

LLM 推理领域发展非常快，最新代码可能包含尚未发布的修复、性能优化和新功能。为了让用户可以随时体验最新代码，无需等待下个版本发布，vLLM 每次提交都会为 x86 平台、CUDA 12 的 Linux 系统构建夜间版 wheel 包。

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

??? console "pip"
    ```bash
    pip install -U vllm \
        --pre \
        --extra-index-url https://wheels.vllm.ai/nightly
    ```

    使用 `pip` 时，需加 `--pre` 参数才能安装预发布版本。

##### 安装指定版本（commit）

如果你需要获取历史提交的 wheel 包（比如定位行为变化或性能回退），可以在 URL 中指定 commit hash：

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支上的完整 commit hash
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

这种 `uv` 安装方式适用于 vLLM `v0.6.6` 及以后版本，命令简单易记。`uv` 的一个特点是，`--extra-index-url` 指定的索引优先级高于默认索引，[具体说明见官方文档](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。比如，若最新正式版为 `v0.6.6.post1`，通过指定 `--extra-index-url` 可以安装 `post1` 之前的 commit wheel。而 `pip` 会合并不同索引，只安装最新版本，所以不方便安装开发中的历史版本。

??? note "pip"
    如果你想用 `pip` 安装历史提交的 wheel，需要在 URL 中直接嵌入 commit hash，指定完整 wheel 文件地址：

    ```bash
    export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # 使用主分支上的完整 commit hash
    pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
    ```

    注意这些 wheel 都使用 Python 3.8 ABI（相关说明见 [PEP 425](https://peps.python.org/pep-0425/)），因此**兼容 Python 3.8 及更高版本**。wheel 文件名中的版本号（如 `1.0.0.dev`）只是占位，实际 wheel 的版本信息在元数据中（extra index url 下的 wheel 都有正确版本号）。虽然我们已不再支持 Python 3.8（因为 PyTorch 2.5 已停止支持），但 wheel 依旧用 Python 3.8 ABI 编译，以保持名称一致。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

#### 仅使用 Python 构建（无需编译）

如果你只需要修改 Python 代码，可以直接构建并安装 vLLM，无需编译。使用 `uv pip` 的 [`--editable` 参数](https://docs.astral.sh/uv/pip/packages/#editable-packages)，你对代码的修改会实时生效：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

该命令会执行以下操作：

1. 检查你当前 vLLM 仓库所在分支。
1. 找到与主分支对应的基础 commit。
1. 下载对应 commit 的预编译 wheel 包。
1. 在安装时使用其中的二进制库。

!!! note
    1. 如果你修改了 C++ 或 kernel（内核）代码，则不能使用 Python-only 构建，否则会出现库未找到或符号未定义的导入错误。
    2. 如果你对开发分支进行了 rebase，建议先卸载 vllm，再重新运行上述命令，以确保二进制库是最新的。

如果上述命令提示 wheel 未找到，可能是你所依赖的主分支 commit 刚刚合并，wheel 包还在构建中。可等待一小时后重试，或手动指定前一个 commit，通过设置环境变量 `VLLM_PRECOMPILED_WHEEL_LOCATION` 指定 wheel 包地址：

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支上的完整 commit hash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
uv pip install --editable .
```

更多有关 vLLM wheel 的信息，请参考[安装最新代码](#install-the-latest-code)。

!!! note
    如果你的源码 commit ID 与已安装 vLLM wheel 不一致，可能会导致未知错误。建议源码和 wheel 使用相同 commit ID。具体安装方法请参考[安装最新代码](#install-the-latest-code)。

#### 完全编译安装（包含编译过程）

如果你需要修改 C++ 或 CUDA 代码，则必须从源码编译 vLLM。整个过程会花费几分钟：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e .
```

!!! tip
    从源码编译需要大量时间。如果你经常编译，建议启用编译缓存，以加快速度。

    比如可以用 `conda install ccache` 或 `apt install ccache` 安装 [ccache](https://github.com/ccache/ccache)。只要 `which ccache` 能找到 ccache 二进制，编译系统会自动使用。首次编译后，后续编译会快很多。

    用 `pip install -e .` 时，配合 ccache，建议运行 `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`。因为 pip 每次编译会新建一个随机目录，ccache 可能无法识别重复文件。

    [sccache](https://github.com/mozilla/sccache) 与 ccache 类似，还支持远程存储缓存。可设置如下环境变量配置 vLLM sccache 远程缓存：`SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`，推荐另加 `SCCACHE_IDLE_TIMEOUT=0`。

!!! note "更快的内核开发"
    如果你频繁修改 C++/CUDA 内核代码，首次用 `uv pip install -e .` 安装后，可以采用[增量编译流程](../../contributing/incremental_build.md)，只编译变更的部分，大幅加快重编译速度。

##### 使用已有的 PyTorch 安装

有些场景下无法用 `uv` 方便地安装 PyTorch，例如：

- 需要用 PyTorch nightly 版或自定义 PyTorch 构建
- 在 aarch64 + CUDA (GH200) 平台编译 vLLM，而 PyTorch wheel 尚未在 PyPI 发布。目前仅 PyTorch nightly 版提供 aarch64 + CUDA wheel。可用 `uv pip install --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio` [安装 PyTorch nightly](https://pytorch.org/get-started/locally/)，再编译 vLLM。

用已有的 PyTorch 环境编译 vLLM：

```bash
# 先安装 PyTorch（可用 PyPI 或源码）
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

另外，如果你一直用 `uv` 管理虚拟环境，它有一套[特殊机制](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)可以为指定包禁用构建隔离。vLLM 可利用该机制对 `torch` 禁用隔离：

```bash
# 先安装 PyTorch（可用 PyPI 或源码）
git clone https://github.com/vllm-project/vllm.git
cd vllm
# pip install -e . 直接用不了，需用 uv
uv pip install -e .
```

##### 使用本地 cutlass 编译

目前 vLLM 编译时会自动从 GitHub 拉取 cutlass 代码。有时你可能想用本地的 cutlass 版本，可设置环境变量 VLLM_CUTLASS_SRC_DIR 指向本地 cutlass 目录：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass uv pip install -e .
```

##### 编译故障排查

为避免系统负载过高，可用环境变量 `MAX_JOBS` 限制同时编译的任务数，比如：

```bash
export MAX_JOBS=6
uv pip install -e .
```

这在性能较弱的机器上尤其有用。例如 WSL 默认只分配 50% 内存（[官方说明](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings)），可用 `export MAX_JOBS=1` 限制只编译一个文件，避免内存耗尽。副作用是编译速度会明显变慢。

此外，如果编译 vLLM 遇到问题，建议直接用 NVIDIA 官方 PyTorch Docker 镜像：

```bash
# 加上 `--ipc=host`，确保共享内存足够
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

如果不想用 Docker，建议完整安装 CUDA Toolkit。可从[官方页面](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装。安装后，设置环境变量 `CUDA_HOME` 指向 CUDA Toolkit 路径，并确保 `nvcc` 编译器已加入 `PATH`，例如：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

安装完毕后，可通过以下方式检查 CUDA Toolkit 是否正常：

```bash
nvcc --version # 检查 nvcc 是否在 PATH 中
${CUDA_HOME}/bin/nvcc --version # 检查 nvcc 是否在 CUDA_HOME 路径下
```

#### 不支持的系统编译

vLLM 仅能在 Linux 上完整运行，但在其他系统（如 macOS）也可用于开发，比如可以正常导入模块，方便调试，但二进制文件不会被编译，不能在非 Linux 系统运行。

只需在安装前禁用 `VLLM_TARGET_DEVICE` 环境变量即可：

```bash
export VLLM_TARGET_DEVICE=empty
uv pip install -e .
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

关于如何使用官方 Docker 镜像，请参考[Docker 使用方法](../../deployment/docker.md)。

另一种获取最新代码的方式是使用 docker 镜像：

```bash
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # 使用主分支上的完整 commit hash
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}
```

这些 docker 镜像仅用于持续集成和测试，并不适用于生产环境。镜像会在几天后自动过期。

请注意，最新代码可能存在 bug，稳定性无法保证，请谨慎使用。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

关于如何从源码构建 vLLM 的 Docker 镜像，请参考[构建方法](../../deployment/docker.md#building-vllms-docker-image-from-source)。

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

关于功能支持和硬件兼容性，请查阅[功能 x 硬件支持矩阵](../../features/README.md#feature-x-hardware)。

# --8<-- [end:supported-features]
