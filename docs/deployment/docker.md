# 使用 Docker

## 使用 vLLM 官方 Docker 镜像

vLLM 提供了官方的 Docker 镜像，方便部署使用。
这个镜像可以用于运行兼容 OpenAI 的服务端，并已在 Docker Hub 上发布：[vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags)

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

这个镜像同样可以搭配其他容器引擎使用，比如 [Podman](https://podman.io/)。

```bash
podman run --device nvidia.com/gpu=all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

你可以在镜像标签（如 `vllm/vllm-openai:latest`）后面，添加你需要的其他 [engine-args](../configuration/engine_args.md)。

!!! note
    你可以选择使用 `ipc=host` 参数或 `--shm-size` 参数，以允许容器访问主机的共享内存。vLLM 使用 PyTorch，而 PyTorch 在底层会使用共享内存来在多个进程间共享数据，尤其是在进行张量并行推理时。

!!! note
    镜像中未包含可选依赖项，以避免相关的许可问题（如 <https://github.com/vllm-project/vllm/issues/8030>）

    如果你需要使用这些可选依赖（并且你已同意相关许可条款），可以基于基础镜像自定义 Dockerfile，在其上增加安装所需依赖的新层：

    ```Dockerfile
    FROM vllm/vllm-openai:v0.11.0

    # 例如，安装 `audio` 可选依赖
    # 注意：请确保 vLLM 的版本与你的基础镜像一致！
    RUN uv pip install --system vllm[audio]==0.11.0
    ```

!!! tip
    有些新模型可能只在 [HF Transformers](https://github.com/huggingface/transformers) 的主分支上才可以使用。

    如果你需要使用开发版的 `transformers`，可以基于官方基础镜像自定义 Dockerfile，在其上直接从源码安装：

    ```Dockerfile
    FROM vllm/vllm-openai:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

## 从源码构建 vLLM Docker 镜像

你可以通过项目提供的 [docker/Dockerfile](../../docker/Dockerfile) 从源码构建并运行 vLLM。构建命令如下：

```bash
# 可选参数：--build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

!!! note
    默认情况下，vLLM 会为所有 GPU 类型进行构建，以便适配更广泛的硬件。如果你只需为当前机器的 GPU 类型构建，可以添加参数 `--build-arg torch_cuda_arch_list=""`，让 vLLM 自动检测当前 GPU 类型并为其构建。

    如果你使用 Podman 而非 Docker，建议在执行 `podman build` 时添加 `--security-opt label=disable`，以避免某些已知 [问题](https://github.com/containers/buildah/discussions/4184)。

## 针对 Arm64/aarch64 构建

你可以为 aarch64 架构（如 Nvidia Grace-Hopper）构建 docker 容器。目前，这需要使用 PyTorch Nightly 版本，并且属于**实验性功能**。通过添加 `--platform "linux/arm64"` 参数即可尝试为 arm64 构建。

!!! note
    由于需要编译多个模块，整个构建过程可能会耗时较长。建议搭配 `--build-arg max_jobs=` 与 `--build-arg nvcc_threads=` 参数来加速构建过程。但要确保 `max_jobs` 的值远大于 `nvcc_threads`，这样效果更佳。并且注意多任务并行时的内存占用可能会非常大（见下方示例）。

??? console "命令示例"

    ```bash
    # 下面是在 Nvidia GH200 服务器上的构建示例。（内存占用约 15GB，构建时长约 1475 秒 / 25 分钟，镜像大小 6.93GB）
    python3 use_existing_torch.py
    DOCKER_BUILDKIT=1 docker build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t vllm/vllm-gh200-openai:latest \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX"
    ```

!!! note
    如果你在非 ARM 主机上（如 x86_64 机器）构建 `linux/arm64` 镜像，需要确保系统已配置好 QEMU 以实现交叉编译，这样主机才能模拟 ARM64 的运行环境。

    请在主机上执行以下命令来注册 QEMU user static 处理器：

    ```bash
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

    完成 QEMU 配置后，即可在 `docker build` 命令中使用 `--platform "linux/arm64"` 参数了。

## 使用自定义构建的 vLLM Docker 镜像

使用自定义构建的 Docker 镜像运行 vLLM：

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

这里的 `vllm/vllm-openai` 表示要运行的镜像标签，你可以替换成你构建时自定义的镜像名（即 build 命令中的 `-t` 标签）。

!!! note
    **仅限 0.4.1 和 0.4.2 版本** —— 这两个版本的 vLLM docker 镜像需要以 root 用户身份运行，因为运行时需要加载 root 用户家目录下的一个库文件，即 `/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`。如果你要用其他用户运行容器，请先修改该库文件（及其父目录）的权限，使其能被该用户访问，然后运行 vLLM 时通过环境变量 `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` 指定库路径。