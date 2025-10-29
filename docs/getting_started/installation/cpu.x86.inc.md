# --8<-- [start:installation]

vLLM 支持在 x86 CPU 平台上进行基础的模型推理和服务，支持的数据类型包括 FP32、FP16 和 BF16。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：Linux
- CPU 指令集：`avx512f`（推荐），`avx512_bf16`（可选），`avx512_vnni`（可选）

!!! tip
    可以使用 `lscpu` 命令来查看 CPU 的指令集支持情况。

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

安装推荐的编译器。我们建议将 `gcc/g++ >= 12.3.0` 作为默认编译器，以避免潜在的问题。例如，在 Ubuntu 22.4 上可以执行：

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

克隆 vLLM 项目代码：

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

安装所需依赖包：

```bash
uv pip install -r requirements/cpu-build.txt --torch-backend cpu
uv pip install -r requirements/cpu.txt --torch-backend cpu
```

??? console "pip"
    ```bash
    pip install --upgrade pip
    pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

构建并安装 vLLM：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

如果需要开发 vLLM，可使用可编辑模式进行安装：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

可选：构建一个可移植的安装包（wheel），方便在其他环境中安装：

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel
```

```bash
uv pip install dist/*.whl
```

??? console "pip"
    ```bash
    VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
    ```

    ```bash
    pip install dist/*.whl
    ```

!!! example "常见问题排查"
    - **NumPy ≥2.0 报错**：可通过 `pip install "numpy<2.0"` 降级解决。
    - **CMake 误检测到 CUDA**：在进行 CPU 构建时，即使已安装 CUDA，也可通过添加 `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` 避免检测到 CUDA。
    - `AMD` 平台要求至少 4 代处理器（Zen 4/Genoa 及以上），才能支持 [AVX512](https://www.phoronix.com/review/amd-zen4-avx512)，以在 CPU 上运行 vLLM。
    - 如果出现类似 `Could not find a version that satisfies the requirement torch==X.Y.Z+cpu+cpu` 的报错，可尝试更新 [pyproject.toml](https://github.com/vllm-project/vllm/blob/main/pyproject.toml)，有助于 pip 解决依赖问题。
    ```toml title="pyproject.toml"
    [build-system]
    requires = [
      "cmake>=3.26.1",
      ...
      "torch==X.Y.Z+cpu"   # <-------
    ]
    ```
    - 如果你是从源码构建 vLLM，而没有使用官方预构建镜像，记得在 x86 机器运行 vLLM 前设置环境变量：`LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD"`。

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

[https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo](https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo)

!!! warning
    如果你在没有 `avx512f`、`avx512_bf16` 或 `avx512_vnni` 支持的机器上部署预构建镜像，可能会遇到 `Illegal instruction` 错误。建议针对这些设备，构建镜像时使用合适的构建参数（例如 `--build-arg VLLM_CPU_DISABLE_AVX512=true`，`--build-arg VLLM_CPU_AVX512BF16=false` 或 `--build-arg VLLM_CPU_AVX512VNNI=false`），以禁用不支持的特性。注意，如果没有 `avx512f`，将会使用 AVX2，但该版本只具备基础功能，不推荐生产环境使用。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512BF16=false (默认值)|true \
        --build-arg VLLM_CPU_AVX512VNNI=false (默认值)|true \
        --build-arg VLLM_CPU_DISABLE_AVX512=false (默认值)|true \ 
        --tag vllm-cpu-env \
        --target vllm-openai .

# 启动 OpenAI 服务
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV 缓存空间> \
            -e VLLM_CPU_OMP_THREADS_BIND=<用于推理的 CPU 核心数> \
            vllm-cpu-env \
            --model=meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            其他 vLLM OpenAI 服务参数
```

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]