# --8<-- [start:installation]

vLLM 已经对 IBM Z 平台上的 s390x 架构提供了实验性支持。当前，用户必须从源码编译，才能在 IBM Z 平台上原生运行。

目前，针对 s390x 架构的 CPU 实现仅支持 FP32 数据类型。

!!! warning
    目前没有为此设备提供预构建的 wheels 或镜像，因此您必须从源码编译 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：`Linux`
- 开发工具：`gcc/g++ >= 12.3.0` 或更高版本，并配有命令行工具
- 指令集架构（ISA）：必须支持 VXE。适用于 Z14 及以上型号。
- 构建安装所需 Python 包：`pyarrow`、`torch` 和 `torchvision`

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

在编译 vLLM 之前，请使用包管理器安装以下软件包。例如，在 RHEL 9.4 上可以这样操作：

```bash
dnf install -y \
    which procps findutils tar vim git gcc g++ make patch make cython zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel wget autoconf automake libtool cmake numactl-devel
```

安装 rust>=1.80，这是安装 `outlines-core` 和 `uvloop` Python 包所必需的。

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env"
```

执行以下命令即可从源码编译并安装 vLLM。

!!! tip
    请务必在编译 vLLM 之前，先从源码编译好 `torchvision` 和 `pyarrow` 这两个依赖包。

```bash
    sed -i '/^torch/d' requirements/build.txt    # 从 requirements/build.txt 移除 torch，因为我们使用的是 nightly 版本
    uv pip install -v \
        --torch-backend auto \
        -r requirements/build.txt \
        -r requirements/cpu.txt \
    VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
        uv pip install dist/*.whl
```

??? console "pip"
    ```bash
        sed -i '/^torch/d' requirements/build.txt    # 从 requirements/build.txt 移除 torch，因为我们使用的是 nightly 版本
        pip install -v \
            --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
            -r requirements/build.txt \
            -r requirements/cpu.txt \
        VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
            pip install dist/*.whl
    ```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.s390x \
    --tag vllm-cpu-env .

# 启动 OpenAI 服务器
docker run --rm \
    --privileged true \
    --shm-size 4g \
    -p 8000:8000 \
    -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
    -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
    vllm-cpu-env \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dtype float \
    其他 vLLM OpenAI server 参数
```

!!! tip
    除了 `--privileged true` 以外，也可以使用 `--cap-add SYS_NICE --security-opt seccomp=unconfined` 作为替代方案。

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]