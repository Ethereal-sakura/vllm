# --8<-- [start:installation]

vLLM 已适配运行在支持 NEON 的 ARM64 架构 CPU 上，底层利用了最初为 x86 平台开发的 CPU 后端。

目前，ARM CPU 后端支持 Float32、FP16 和 BFloat16 三种数据类型。

!!! warning
    目前没有为该设备提供预编译的 wheel 或镜像包，因此你需要从源码自行构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：Linux
- 编译器：`gcc/g++ >= 12.3.0`（可选，但推荐使用）
- 指令集架构（ISA）：需要支持 NEON

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

首先，安装推荐的编译器。我们建议将 `gcc/g++ >= 12.3.0` 设置为默认编译器，以避免潜在的问题。例如，在 Ubuntu 22.4 上可以执行：

```bash
sudo apt-get update  -y
sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

然后，克隆 vLLM 项目源码：

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

接下来，安装所需依赖项：

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

最后，构建并安装 vLLM：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

如果你希望进行二次开发，可以使用可编辑模式安装：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

在 AWS Graviton3 实例上已进行兼容性测试。

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]
```bash
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env .

# 启动 OpenAI 服务器
docker run --rm \
            --privileged=true \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-env \
            --model=meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

!!! tip
    你也可以使用 `--cap-add SYS_NICE --security-opt seccomp=unconfined` 作为 `--privileged=true` 的替代方案。

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
