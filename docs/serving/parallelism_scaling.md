# 并行化与扩展

## 单模型副本的分布式推理策略

选择单模型副本的分布式推理策略时，可参考以下建议：

- **单 GPU（无分布式推理）：** 如果模型能够完全加载到一张 GPU 上，则无需使用分布式推理。直接在该 GPU 上运行推理即可。
- **单节点多 GPU，使用张量并行推理：** 如果模型太大，单张 GPU 放不下，但在一台有多张 GPU 的节点上可以容纳，则可使用 *张量并行（tensor parallelism）*。例如，若节点有 4 张 GPU，设置 `tensor_parallel_size=4`。
- **多节点多 GPU，结合张量并行和流水线并行推理：** 如果模型单节点也装不下，可以把 *张量并行* 和 *流水线并行（pipeline parallelism）* 结合起来。`tensor_parallel_size` 设为每个节点的 GPU 数，`pipeline_parallel_size` 设为节点数。例如，2 个节点，每节点 8 张 GPU，设置 `tensor_parallel_size=8`，`pipeline_parallel_size=2`。

根据模型大小，逐步增加 GPU 和节点数，直到 GPU 显存足够容纳模型。`tensor_parallel_size` 设置为每节点 GPU 数，`pipeline_parallel_size` 设置为节点总数。

当资源分配足够后，运行 `vllm`，可在日志中看到类似如下信息：

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

`GPU KV cache size` 一行表示当前 GPU KV 缓存最多可存储的 token 数量。`Maximum concurrency` 则估算了当每次请求需要指定数量的 token（如上述例子为 40,960）时，最多可同时服务多少个请求。tokens-per-request 的数量取自模型配置里的最大序列长度 `ModelConfig.max_model_len`。如果这些数据低于你的吞吐需求，可以扩展更多 GPU 或节点。

!!! note "特殊情况：GPU 分割不均"
    如果模型可以放在单节点，但 GPU 数不能均匀分割模型，可以启用流水线并行（pipeline parallelism），它会按层切分模型，支持不均匀分割。这种情况下，将 `tensor_parallel_size` 设为 1，`pipeline_parallel_size` 设为 GPU 数量。此外，如果节点上的 GPU 没有 NVLINK 互联（例如 L40S），建议使用流水线并行而不是张量并行，以获得更高吞吐量和更低通信开销。

### 分布式部署 *专家混合模型*（Mixture of Experts，MoE）

在部署时，合理利用专家层（expert layers）的并行特性非常重要。vLLM 支持将数据并行注意力（Data Parallel attention）与专家或张量并行 MoE 层组合，实现大规模部署。详情请参考 [Data Parallel Deployment](data_parallel_deployment.md)。

## 单节点部署

vLLM 支持张量并行和流水线并行的分布式推理与服务。实现中包含了 [Megatron-LM 的张量并行算法](https://arxiv.org/pdf/1909.08053.pdf)。

默认的分布式运行时为多节点推理使用 [Ray](https://github.com/ray-project/ray)，单节点推理使用原生 Python 的 `multiprocessing`。你可以通过设置 `LLM` 类的 `distributed_executor_backend` 或 API 服务中的 `--distributed-executor-backend` 参数来切换默认执行器。`mp` 表示使用 `multiprocessing`，`ray` 表示使用 Ray。

多 GPU 推理时，在 `LLM` 类中设置 `tensor_parallel_size` 为所需 GPU 数。例如，使用 4 张 GPU 运行推理：

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

多 GPU 服务时，启动服务时添加 `--tensor-parallel-size` 参数。例如，API 服务运行在 4 张 GPU：

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

如需启用流水线并行，可添加 `--pipeline-parallel-size` 参数。例如，API 服务在 8 张 GPU 上同时启用张量并行和流水线并行：

```bash
# 总计 8 张 GPU
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## 多节点部署

如果单节点 GPU 数不足以加载模型，可以将 vLLM 部署在多节点集群。务必确保每个节点的运行环境一致，包括模型路径和 Python 包。建议使用容器镜像，这样能方便地保持环境一致性并屏蔽宿主机差异。

### Ray 是什么？

Ray 是一个专为 Python 程序扩展而设计的分布式计算框架。多节点 vLLM 部署需要 Ray 作为运行引擎。

vLLM 通过 Ray 管理任务在各节点上的分布式执行，并灵活控制任务调度。

Ray 还提供了大规模 [离线批量推理](https://docs.ray.io/zh/latest/data/working-with-llms.html) 和 [在线服务](https://docs.ray.io/zh/latest/serve/llm) 的高级 API，这些 API 可直接调用 vLLM 作为底层引擎。它们为 vLLM 任务带来了生产级容错、自动扩展与分布式监控等能力。

更多内容请查阅 [Ray 官方文档](https://docs.ray.io/zh/latest/index.html)。

### 使用容器一键部署 Ray 集群

辅助脚本 [examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh) 可在多节点启动容器并初始化 Ray。默认情况下，该脚本运行 Docker 时没有管理员权限，这样在性能分析或追踪时无法访问 GPU 性能计数器。如需管理员权限，请在 Docker 命令中添加 `--cap-add=CAP_SYS_ADMIN` 参数。

选择一个节点作为 head 节点，运行以下命令：

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<HEAD_NODE_IP>
```

在每个 worker 节点上运行：

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

注意，每个 worker 的 `VLLM_HOST_IP` 都需唯一。运行这些命令的 shell 需保持开启，任何一个 shell关闭都会导致集群停止。确保所有节点之间可以通过 IP 相互访问。

!!! warning "网络安全提醒"
    为了安全起见，建议将 `VLLM_HOST_IP` 设为私有网段地址。该网络上的数据传输为明文，且端点间的数据格式可能被恶意利用从而执行任意代码。如果网络被攻击者访问，风险较高。务必保证不可信人员无法访问该网络。

在任意节点进入容器后，运行 `ray status` 和 `ray list nodes` 检查 Ray 是否识别了预期的节点和 GPU 数量。

!!! tip
    你也可以通过 KubeRay 部署 Ray 集群。详细信息请参考 [KubeRay vLLM 文档](https://docs.ray.io/zh/latest/cluster/kubernetes/examples/rayserve-llm-example.html)。

### 在 Ray 集群上运行 vLLM

!!! tip
    如果 Ray 在容器内运行，后续所有命令都应在容器内执行而不是主机上。进入容器可通过 `docker exec -it <container_name> /bin/bash`。

Ray 集群启动后，使用 vLLM 的方式与单节点类似。vLLM 会自动识别整个 Ray 集群中的所有资源，因此只需在任一节点执行一次 `vllm` 命令即可。

通常做法是将张量并行规模设为每节点 GPU 数，流水线并行规模设为节点总数。例如，2 个节点共 16 张 GPU（每节点 8 张），则张量并行规模为 8，流水线并行规模为 2：

```bash
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

你也可以将 `tensor_parallel_size` 设置为整个集群总 GPU 数：

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16
```

## 优化张量并行的网络通信

高效的张量并行需要节点间高速网络通信，建议采用如 InfiniBand 等高性能网卡。
如需配置集群使用 InfiniBand，可在
[examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh) 辅助脚本中添加 `--privileged -e NCCL_IB_HCA=mlx5` 等参数。
具体参数请联系你的系统管理员。

## 启用 GPUDirect RDMA

GPUDirect RDMA（远程直接内存访问，Remote Direct Memory Access）是 NVIDIA 推出的技术，允许网卡直接访问 GPU 显存，无需经过 CPU 和系统内存。这种直接访问可降低延迟和 CPU 占用，对于节点间大数据量的 GPU 互传非常有益。

如需在 vLLM 中启用 GPUDirect RDMA，需配置如下：

- `IPC_LOCK` 安全上下文：为容器添加 `IPC_LOCK` 权限，以锁定内存页面防止被换出到磁盘。
- 挂载 `/dev/shm` 共享内存：在 pod 配置中挂载 `/dev/shm`，用于进程间通信（IPC）共享内存。

若使用 Docker，可按如下方式启动容器：

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

若使用 Kubernetes，可按如下方式配置 pod：

```yaml
...
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
...
```

!!! tip "如何确认 GPUDirect RDMA 工作状态"
    想确认你的 InfiniBand 网卡是否启用 GPUDirect RDMA，可用详细 NCCL 日志运行 vLLM：`NCCL_DEBUG=TRACE vllm serve ...`

    日志会显示 NCCL 版本和实际使用的网络类型。

    - 若日志中出现 `[send] via NET/IB/GDRDMA`，说明 NCCL 正在用 InfiniBand + GPUDirect RDMA，效率较高。
    - 若出现 `[send] via NET/Socket`，说明 NCCL 用的是普通 TCP socket，跨节点张量并行效率较低。

!!! tip "提前下载 Hugging Face 模型"
    如果使用 Hugging Face 模型，建议提前下载模型。可在每个节点下载到相同路径，或统一存放在全节点可访问的分布式文件系统。启动 vLLM 时传递模型路径即可，无需 repo id。否则，也可通过在 `run_cluster.sh` 后追加 `-e HF_TOKEN=<TOKEN>` 传递 Hugging Face token。

## 分布式部署故障排查

分布式调试相关信息请参考 [分布式部署故障排查](distributed_troubleshooting.md)。
