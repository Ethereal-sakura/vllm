# 数据并行部署

vLLM 支持数据并行（Data Parallel）部署方式，即在多个独立实例或 GPU 上复制模型权重，从而能够并行处理互不相关的请求批次。

这种方式同时支持普通密集模型（dense models）和 MoE（Mixture of Experts，专家混合）模型。

对于 MoE 模型，尤其是像 DeepSeek 这类采用 MLA（Multi-head Latent Attention，多头潜在注意力）的模型，推荐将注意力层采用数据并行，而专家层则使用专家并行（Expert Parallel，EP）或张量并行（Tensor Parallel，TP）。

在这种场景下，各个数据并行 rank 之间并不是完全独立的。前向推理时需要保持同步，并且所有 rank 的专家层需要在每次前向计算时都同步，即使当前待处理的请求数量少于 DP rank 数时也需要同步。

专家层默认会组成一个 (DP x TP) 大小的张量并行组。如果要开启专家并行，请在所有节点上添加 `--enable-expert-parallel` 启动参数。

在 vLLM 中，每个 DP rank 会作为独立的“核心引擎”进程运行，并通过 ZMQ socket 与前端进程通信。数据并行的注意力层可以与张量并行结合使用，此时每个 DP 引擎下会启动数量与 TP 大小相同的每 GPU worker 进程。

针对 MoE 模型，如果某个 rank 上有请求在处理中，则需要确保其他没有请求的 rank 也执行空的“虚拟”前向计算。这一过程由专门的 DP 协调器进程负责，它会与所有 rank 通信，并定期进行一次全局协作操作，以判断所有 rank 是否都空闲，可以暂停计算。当 DP 与 TP 同时使用时，专家层会组成 (DP x TP) 大小的 EP 或 TP 并行组。

无论哪种情况，将请求负载均衡到各个 DP rank 都有助于提升效率。在在线部署场景下，负载均衡可以根据每个 DP 引擎的当前状态（包括已调度和排队等待的请求，以及 KV cache 状态）进行优化。每个 DP 引擎拥有独立的 KV cache，通过智能分配 prompt，可以最大化前缀缓存带来的性能提升。

本文档主要介绍在线部署（即配合 API server）的场景。DP + EP 也支持离线使用（通过 LLM 类），相关示例可参考 [examples/offline_inference/data_parallel.py](../../examples/offline_inference/data_parallel.py)

在线部署支持两种模式：一种是自带内部负载均衡的“一体化”部署，另一种是每个 rank 独立进程部署，由外部实现负载均衡。

## 内部负载均衡

vLLM 支持“一体化”数据并行部署方式，仅需暴露一个 API 端点。

只需在 vllm serve 的命令行参数中添加 `--data-parallel-size=4`，即可配置 4 卡并行。也可以与张量并行结合，例如 `--data-parallel-size=4 --tensor-parallel-size=2`，此时需要 8 张 GPU。

如果需要跨多节点部署单个数据并行实例，需要在每个节点上分别运行 `vllm serve`，并指定该节点上运行的 DP rank。此时，HTTP 入口依然只有一个——API server 只需运行在其中一个节点上，也不必和 DP rank 进程部署在同一台机器。

以下示例是在单台 8 卡服务器上运行 DP=4, TP=2：

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

以下示例将 DP=4 的 rank 0 和 1 部署在主节点，rank 2 和 3 部署在第二台节点：

```bash
# 节点 0  (ip 地址为 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# 节点 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

以下示例将 API server 仅部署在第一台节点，所有引擎进程都部署在第二台节点：

```bash
# 节点 0  (ip 地址为 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# 节点 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

这种数据并行模式也可以结合 Ray 使用，只需指定 `--data-parallel-backend=ray`：

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

使用 Ray 部署时有以下特点：

- 只需要在任意节点上执行一次启动命令，即可同时启动所有本地和远程的 DP rank，使用上比在每个节点分别启动更为简便
- 无需再指定 `--data-parallel-address`，命令运行所在节点自动作为 `--data-parallel-address`
- 无需指定 `--data-parallel-rpc-port`
- 当单个 DP 组需要跨多个节点部署（比如一个模型副本需要在至少两台节点上运行）时，请确保设置环境变量 `VLLM_RAY_DP_PACK_STRATEGY="span"`，此时 `--data-parallel-size-local` 会被忽略，由系统自动分配
- Ray 集群会根据各节点资源自动为远程 DP rank 分配资源

目前，内部的数据并行负载均衡是在 API server 进程内部实现的，调度依据是每个引擎的运行队列和等待队列。未来也可能加入更智能的 KV cache 感知调度逻辑。

当采用这种方式部署大规模 DP 时，API server 进程可能会成为瓶颈。这时可以通过 `--api-server-count` 参数横向扩展 API server（例如 `--api-server-count=4`）。对用户来说，这种扩展是透明的，依然只需访问一个 HTTP 端点/端口。需要注意的是，这种 API server 扩展仅限于同一个“主”节点内部。

<figure markdown="1">
![DP Internal LB Diagram](../assets/deployment/dp_internal_lb.png)
</figure>

## 外部负载均衡

对于大规模部署场景，将数据并行 rank 的调度和负载均衡交由外部系统管理会更灵活高效。

在这种模式下，可以把每个 DP rank 看作独立的 vLLM 部署实例，分别暴露不同的端口，通过外部路由器（如负载均衡器）将 HTTP 请求分发到各个实例，并结合服务器的实时状态进行智能路由。

对于非 MoE 模型，这种方式非常简单，因为各个服务实例是完全独立的，无需额外的数据并行 CLI 参数。

对于 MoE DP+EP 部署，也支持类似的部署拓扑，只需通过下述命令行参数进行配置即可。

如果 DP rank 部署在同一台节点（同一个 IP 地址），默认 RPC 端口会自动分配，但每个 rank 需指定不同的 HTTP 端口：

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

对于多节点部署，rank 0 的地址和端口需要手动指定：

```bash
# Rank 0  (ip 地址为 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

在这种部署方式下，协调器进程同样会运行，并与 rank 0 的引擎进程部署在一起。

<figure markdown="1">
![DP External LB Diagram](../assets/deployment/dp_external_lb.png)
</figure>

如上图所示，每一个虚线框代表一次独立的 `vllm serve` 启动实例——这些实例可以部署在不同的 Kubernetes pod 上。