# --8<-- [start:installation]

vLLM 目前已支持在 Intel GPU 平台上进行基础的模型推理和服务。

!!! warning
    目前没有针对该设备的预编译安装包（wheels），所以你需要从源码编译 vLLM。或者，你也可以使用基于 vLLM 官方发布版本的预构建镜像。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 支持的硬件：Intel 数据中心 GPU、Intel ARC GPU
- oneAPI 要求：oneAPI 2025.1
- Python：3.12
!!! warning
    提供的 IPEX 安装包仅适用于 Python3.12，因此必须使用这个版本。

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

针对该设备，暂无额外的 Python 环境创建说明。

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

目前还没有提供预编译的 XPU 安装包（wheels）。

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

- 首先，安装所需的 [驱动](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers) 和 [Intel OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) 2025.1 及以上版本。
- 然后，安装用于构建 vLLM XPU 后端的 Python 包：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- 接下来，编译并安装 vLLM XPU 后端：

```bash
VLLM_TARGET_DEVICE=xpu python setup.py install
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

目前，我们基于 vLLM 官方发布版本在 docker [hub](https://hub.docker.com/r/intel/vllm/tags) 提供了预构建的 XPU 镜像。更多信息可参考发布 [说明](https://github.com/intel/ai-containers/blob/main/vllm)。

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

XPU 平台支持 **张量并行（tensor parallel）** 的推理和服务，同时也支持 **流水线并行（pipeline parallel）**，该功能目前处于测试阶段，可用于在线服务。对于流水线并行，我们支持在单节点上以 mp 作为后端运行。例如，参考如下执行命令：

```bash
vllm serve facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=mp \
     --pipeline-parallel-size=2 \
     -tp=8
```

默认情况下，如果系统中没有检测到已有的 ray 实例，会自动启动一个 ray，`num-gpus` 会设为 `parallel_config.world_size`。建议你在运行前自行启动 ray 集群，具体可参考 [examples/online_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh) 脚本。

# --8<-- [end:supported-features]
# --8<-- [start:distributed-backend]

XPU 平台在 torch<2.8 时使用 **torch-ccl** 作为分布式后端，而在 torch>=2.8 时则采用 **xccl**，因为 torch 2.8 及以上版本已内置支持 XPU 的 xccl 后端。

# --8<-- [end:distributed-backend]
