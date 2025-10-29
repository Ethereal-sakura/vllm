# --8<-- [start:installation]

vLLM 支持搭载 ROCm 6.3 及以上版本的 AMD GPU，同时要求 torch 2.8.0 及以上版本。

!!! tip
    推荐通过 [Docker](#set-up-using-docker) 在 ROCm 环境下使用 vLLM。

!!! warning
    目前没有为该设备提供预编译的 wheel 包，因此你需要使用官方预构建的 Docker 镜像，或者从源码自行编译 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU：MI200s (gfx90a)、MI300 (gfx942)、MI350 (gfx950)、Radeon RX 7900 系列 (gfx1100/1101)、Radeon RX 9000 系列 (gfx1200/1201)
- ROCm 6.3 或更高版本
    - MI350 需要 ROCm 7.0 或以上版本

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

对于该设备，无需额外说明如何新建 Python 环境。

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前暂未提供 ROCm 的预编译 wheel 包。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

!!! tip
    - 如果你发现以下安装步骤无法正常工作，请参考 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 。Dockerfile 实际上就是一份详细的安装步骤说明。

0. 安装前置依赖（如果你已经在包含以下组件的环境/docker 中，可以跳过此步）：

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    安装 PyTorch 时，可以直接从官方 docker 镜像开始，例如 `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`、`rocm/pytorch-nightly` 等。如果你使用的是 docker 镜像，可直接跳到第 3 步。

    另外，也可以通过 PyTorch wheel 包手动安装。安装方法可参考 [PyTorch 官方指南](https://pytorch.org/get-started/locally/)。例如：

    ```bash
    # 安装 PyTorch
    pip uninstall torch -y
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
    ```

1. 安装 [Triton for ROCm](https://github.com/ROCm/triton.git)

    按照 [ROCm/triton](https://github.com/ROCm/triton.git) 的官方说明安装 ROCm 版本的 Triton：

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    # git checkout $TRITON_BRANCH
    git checkout f9e5bf54
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```

    !!! note
        - 已验证的 `$TRITON_BRANCH` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。
        - 如果在编译 triton 时遇到下载包的 HTTP 问题，请多尝试几次，这类错误通常是偶发的。

2. 可选：如需使用 CK flash attention，可安装 [flash attention for ROCm](https://github.com/Dao-AILab/flash-attention.git)

    参考 [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support) 安装 ROCm 版 flash attention（v2.8.0）。

    例如，假如你使用 ROCm 7.0，且显卡架构为 `gfx942`。可通过命令 `rocminfo |grep gfx` 查询显卡架构。

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    # git checkout $FA_BRANCH
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS="gfx942" python3 setup.py install
    cd ..
    ```

    !!! note
        - 已验证的 `$FA_BRANCH` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。

3. 如果你希望自行选择分支或提交构建 AITER，可按以下步骤构建：

    ```bash
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        - 请根据实际需求配置 `$AITER_BRANCH_OR_COMMIT`。
        - 已验证的 `$AITER_BRANCH_OR_COMMIT` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。
        

4. 构建 vLLM。例如，在 ROCM 7.0 环境下可按如下步骤安装 vLLM：

    ???+ console "命令示例"

        ```bash
        pip install --upgrade pip

        # 构建并安装 AMD SMI
        pip install /opt/rocm/share/amd_smi

        # 安装依赖
        pip install --upgrade numba \
            scipy \
            huggingface-hub[cli,hf_transfer] \
            setuptools_scm
        pip install -r requirements/rocm.txt

        # 推荐：仅为单一架构（如 MI300）构建以加快安装速度
        export PYTORCH_ROCM_ARCH="gfx942"

        # 如需同时支持 MI210/MI250/MI300 可使用
        # export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

        python3 setup.py develop
        ```

    这一过程大约需要 5-10 分钟。目前，使用 `pip install .` 的方式不支持 ROCm 环境。

    !!! tip
        - 建议 PyTorch 的 ROCm 版本与 ROCm 驱动版本保持一致，以保证兼容性。

!!! tip
    - MI300x（gfx942）用户如需获得最佳性能，可参考 [MI300x 性能调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)，获取系统及流程级别的优化建议。
      vLLM 相关优化建议请参考 [vLLM 性能优化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)。

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

[AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) 提供了官方预编译、性能优化的 docker 镜像，专为在 AMD Instinct™ MI300X 加速卡上进行推理性能验证而设计。
除此之外，AMD 还在 [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) 提供了每日构建的预编译镜像，已预装好 vLLM 及其所有依赖。

???+ console "命令示例"
    ```bash
    docker pull rocm/vllm-dev:nightly # 拉取最新镜像
    docker run -it --rm \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v <path/to/your/models>:/app/models \
    -e HF_HOME="/app/models" \
    rocm/vllm-dev:nightly
    ```

!!! tip
    有关如何使用此预编译 docker 镜像进行推理性能验证，请参阅 [在 AMD Instinct MI300X 上进行 LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

建议通过源码构建 Docker 镜像来在 ROCm 环境中使用 vLLM。

??? info "(可选) 构建包含 ROCm 软件栈的镜像"

    可以使用 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 构建 docker 镜像，预装 vLLM 所需的 ROCm 软件栈。
    **此步骤为可选，通常官方已在 [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) 以 `rocm/vllm-dev:base` 标签预构建并发布该镜像，能大幅缩短部署时间。**
    如果你选择自行构建 rocm_base 镜像，参考下述步骤。

    构建镜像时，务必启用 buildkit。可以在构建命令前加上 DOCKER_BUILDKIT=1 环境变量，或在 /etc/docker/daemon.json 中启用 buildkit，并重启 docker 服务：

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    若需在 ROCm 7.0 环境下为 MI200 和 MI300 系列显卡构建镜像，可直接使用默认命令：

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

#### 构建包含 vLLM 的镜像

首先，使用 [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) 构建 docker 镜像，并用该镜像启动容器。
同样建议在构建时启用 buildkit。可以在命令前设置 `DOCKER_BUILDKIT=1`，或者在 /etc/docker/daemon.json 中启用 buildkit：

```bash
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) 默认基于 ROCm 7.0，但在早期 vLLM 分支中也支持 ROCm 5.7、6.0、6.1、6.2、6.3、6.4。
支持以下参数自定义 docker 镜像构建：

- `BASE_IMAGE`：指定构建时的基础镜像。默认 `rocm/vllm-dev:base`，该镜像由 AMD 官方维护，并基于 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 构建。
- `ARG_PYTORCH_ROCM_ARCH`：可覆盖基础镜像中设置的 gfx 架构参数。

可通过 `docker build` 的 `--build-arg` 选项传递这些参数。

若需在 ROCm 7.0 环境下为 MI200 和 MI300 系列构建 vLLM 镜像，可直接使用默认命令：

???+ console "命令示例"
    ```bash
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm .
    ```

如需运行上面构建的 `vllm-rocm` 镜像，可用如下命令启动：

???+ console "命令示例"
    ```bash
    docker run -it \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v <path/to/model>:/app/model \
    vllm-rocm
    ```

其中 `<path/to/model>` 指向你实际存放模型的位置，例如 llama2 或 llama3 的权重文件目录。

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

关于各项特性支持情况，请参考 [Feature x Hardware](../../features/README.md#feature-x-hardware) 兼容性矩阵。

# --8<-- [end:supported-features]
