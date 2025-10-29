# 专家并行部署

vLLM 支持专家并行（Expert Parallelism，EP），这允许混合专家（Mixture-of-Experts，MoE）模型中的各个专家分别部署在不同的 GPU 上，从而提升局部性、运行效率和整体吞吐量。

EP 通常会与数据并行（Data Parallelism，DP）结合使用。虽然 DP 可以单独启用，但与 DP 配合时 EP 的性能表现更佳。你可以在[这里](data_parallel_deployment.md)了解更多关于数据并行的信息。

## 前置条件

在使用 EP 前，需要安装相关依赖。我们正在持续优化安装流程，未来会更加便捷：

1. **安装 DeepEP 和 pplx-kernels**：请按照 vLLM 的 EP kernels 指南设置主机环境，具体见[这里](../../tools/ep_kernels)。
2. **安装 DeepGEMM 库**：请参考[官方说明](https://github.com/deepseek-ai/DeepGEMM#installation)进行安装。
3. **分离式服务场景**：运行 [`install_gdrcopy.sh`](../../tools/install_gdrcopy.sh) 脚本安装 `gdrcopy`（例如 `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`）。可用操作系统版本见[这里](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/)。

### 通信后端选择指南

vLLM 为 EP 提供了多种通信后端。通过 `--all2all-backend` 参数进行选择：

| 后端 | 适用场景 | 主要特点 | 推荐用途 |
|---------|----------|----------|----------|
| `allgather_reducescatter` | 默认后端 | 标准 all2all，使用 allgather/reducescatter 原语 | 通用，适合任何 EP+DP 配置 |
| `pplx` | 单节点 | 支持分块 prefill，高效节点内通信 | 单节点部署、开发测试 |
| `deepep_high_throughput` | 多节点 prefill | 分组 GEMM、连续布局，prefill 性能优化 | Prefill 密集型、高吞吐场景 |
| `deepep_low_latency` | 多节点 decode | 支持 CUDA 图、掩码布局，decode 性能优化 | Decode 密集型、低延迟场景 |
| `flashinfer_all2allv` | MNNVL 系统 | FlashInfer alltoallv 内核，适用于多节点 NVLink | 跨节点 NVLink 系统 |
| `naive` | 测试/调试 | 简单的广播实现 | 仅限调试，不建议生产使用 |

## 单节点部署

!!! warning
    EP 目前仍为实验性功能，参数命名与默认值未来可能会有所调整。

### 配置方式

启用 EP，只需添加 `--enable-expert-parallel` 参数。EP 的规模会自动计算：

```text
EP_SIZE = TP_SIZE × DP_SIZE
```

其中：

- `TP_SIZE`：张量并行规模（目前始终为 1）
- `DP_SIZE`：数据并行规模
- `EP_SIZE`：专家并行规模（自动计算）

### 命令示例

下面的命令用于部署 `DeepSeek-V3-0324` 模型，采用 1 路张量并行、8 路（注意力）数据并行和 8 路专家并行。注意力权重会在所有 GPU 上复制，专家权重则在 GPU 间分割。适用于 H200（或 H20）节点（8 卡），如果是 H100，可选较小模型或参考多节点部署方式。

```bash
# 单节点 EP 部署，使用 pplx 后端
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # 张量并行（1 卡）
    --data-parallel-size 8 \         # 数据并行（8 进程）
    --enable-expert-parallel \       # 启用专家并行
    --all2all-backend pplx           # 使用 pplx 通信后端
```

## 多节点部署

多节点部署时，需使用 DeepEP 通信内核，并选择两种模式之一（参见前文[通信后端选择指南](#backend-selection-guide)）。

### 部署步骤

1. **每个节点都需单独启动** - 每个节点运行独立的启动命令
2. **设置网络参数** - 确保 IP 地址与端口配置正确
3. **指定节点角色** - 首个节点处理请求，其余节点以 headless 模式运行

### 示例：2 节点部署

以下示例展示如何通过 `deepep_low_latency` 模式，将 `DeepSeek-V3-0324` 部署在 2 个节点上：

```bash
# 节点 1（主节点，负责请求处理）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # 每节点 TP 规模
    --enable-expert-parallel \               # 启用专家并行
    --data-parallel-size 16 \                # 总 DP 规模（所有节点之和）
    --data-parallel-size-local 8 \           # 本节点 DP 规模（8 卡/节点）
    --data-parallel-address 192.168.1.100 \  # 实际主节点 IP
    --data-parallel-rpc-port 13345 \         # 通信端口，需各节点可访问
    --api-server-count=8                     # API server 数量（建议与总进程数一致）

# 节点 2（从节点，仅 worker，无 API server）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # 每节点 TP 规模
    --enable-expert-parallel \               # 启用专家并行
    --data-parallel-size 16 \                # 总 DP 规模
    --data-parallel-size-local 8 \           # 本节点 DP 规模
    --data-parallel-start-rank 8 \           # 本节点的 rank 起始偏移
    --data-parallel-address 192.168.1.100 \  # 主节点 IP
    --data-parallel-rpc-port 13345 \         # 与主节点一致
    --headless                               # 无 API server，仅作为 worker
```

### 配置要点

- **Headless 模式**：从节点需加 `--headless`，所有客户端请求都由主节点处理
- **Rank 计算**：`--data-parallel-start-rank` 设置为前序所有节点的本地 DP 规模之和
- **负载扩展**：主节点的 `--api-server-count` 可根据需求提升并发处理能力

### 网络配置

!!! important "InfiniBand 集群"
    如果是在 InfiniBand 网络集群环境，请设置如下环境变量以避免初始化卡死：
    ```bash
    export GLOO_SOCKET_IFNAME=eth0
    ```
    这样 torch 分布式组的发现过程会优先使用以太网，而不是 InfiniBand。

## 专家并行负载均衡（EPLB）

虽然 MoE 模型训练时会尽量让各专家分配到的 token 数量相近，但实际推理过程中的 token 分布常常十分不均。vLLM 提供了专家并行负载均衡器（Expert Parallel Load Balancer，EPLB），可以在 EP rank 间重新分配专家映射，实现均匀负载。

### 配置方式

通过 `--enable-eplb` 参数启用 EPLB。

!!! note "模型支持情况"
    目前仅支持 DeepSeek V3 架构。

启用后，vLLM 会在每次前向推理时收集专家负载数据，并定期自动优化专家分布。

### EPLB 参数

使用 `--eplb-config` 参数配置 EPLB，传入 JSON 字符串。各可选键及含义如下：

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `window_size`| 用于均衡决策的数据窗口步数 | 1000 |
| `step_interval`| 重新分配的周期（每 N 步） | 3000 |
| `log_balancedness` | 是否记录均衡度（平均分配 token / 最大分配 token） | `false` |
| `num_redundant_experts` | 每个 EP rank 额外增加的全局专家数 | `0` |

例如：

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

??? tip "想用独立参数而不是 JSON？"

    ```bash
    vllm serve Qwen/Qwen3-30B-A3B \
            --enable-eplb \
            --eplb-config.window_size 1000 \
            --eplb-config.step_interval 3000 \
            --eplb-config.num_redundant_experts 2 \
            --eplb-config.log_balancedness true
    ```

### 专家分配公式

- **默认分配**：每个 EP rank 分配到 `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS` 个专家
- **冗余分配**：每个 EP rank 分配到 `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` 个专家

### 显存开销

EPLB 会用到冗余专家，需保证这些专家能全部放入 GPU 显存。如果显存受限或 KV cache 用量较大，EPLB 可能不适合。

开销计算公式为：`NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`。
以 DeepSeekV3 为例，每个 EP rank 多加一个冗余专家约需 `2.4 GB` 显存。

### 命令示例

单节点 EPLB 部署示例：

```bash
# 单节点启用 EPLB 负载均衡
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # 张量并行
    --data-parallel-size 8 \         # 数据并行
    --enable-expert-parallel \       # 启用专家并行
    --all2all-backend pplx \         # pplx 通信后端
    --enable-eplb \                  # 启用负载均衡
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

多节点部署时，请在每个节点命令中添加 EPLB 参数。大规模场景建议设置 `--eplb-config '{"num_redundant_experts":32}'`，确保热点专家总是可用。

## 分离式服务（Prefill/Decode 分离）

在生产环境中，为了提升首 token 响应和多 token 生成的延迟表现，可以采用分离式服务架构，分别独立扩展 prefill 和 decode 处理能力。

### 架构概述

- **Prefill 实例**：采用 `deepep_high_throughput` 后端，优化 prefill 性能
- **Decode 实例**：采用 `deepep_low_latency` 后端，最大限度降低 decode 延迟  
- **KV cache 传递**：通过 NIXL 或其他 KV 连接器实现实例间 KV cache 共享

### 部署步骤

1. **安装 gdrcopy/ucx/nixl**：推荐运行 [install_gdrcopy.sh](../../tools/install_gdrcopy.sh) 脚本安装 `gdrcopy`（例如 `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`）。可用版本见[这里](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/)。如未安装 gdrcopy，也可通过 `pip install nixl` 使用，性能略低。`nixl` 和 `ucx` 可直接 pip 安装。非 cuda 平台安装 nixl 的非 cuda UCX 版本，可运行 [install_nixl_from_source_ubuntu.py](../../tools/install_nixl_from_source_ubuntu.py)。

2. **配置两类实例**：在 prefill 和 decode 实例均添加如下参数：`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}`。此外也可指定多个 NIXL_Backend，例如：`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'`

3. **客户端编排**：可用下方示例脚本协调 prefill/decode 操作。我们正在开发更完善的路由方案。

### 客户端编排示例

```python
from openai import OpenAI
import uuid

try:
    # 1: 分别创建 prefill 和 decode 实例的客户端
    openai_api_key = "EMPTY"  # vLLM 无需真实 API key
    
    # 请将 IP 地址替换为实际实例地址
    prefill_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.100:8000/v1",  # Prefill 实例 URL
    )
    decode_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.101:8001/v1",  # Decode 实例 URL  
    )
    
    # 获取模型名称
    models = prefill_client.models.list()
    model = models.data[0].id
    print(f"使用模型: {model}")

    # 2: Prefill 阶段
    # 生成唯一请求 ID，关联 prefill 与 decode 操作
    request_id = str(uuid.uuid4())
    print(f"请求 ID: {request_id}")
    
    prefill_response = prefill_client.completions.create(
        model=model,
        # Prompt 必须超过 vLLM 的 block size（16 token），PD 才能生效
        prompt="Write a detailed explanation of Paged Attention for Transformers works including the management of KV cache for multi-turn conversations",
        max_tokens=1,  # 强制仅 prefill
        extra_body={
            "kv_transfer_params": {
                "do_remote_decode": True,     # 启用远程 decode
                "do_remote_prefill": False,   # 当前为 prefill 实例
                "remote_engine_id": None,     # vLLM 自动填充
                "remote_block_ids": None,     # vLLM 自动填充
                "remote_host": None,          # vLLM 自动填充
                "remote_port": None,          # vLLM 自动填充
            }
        },
        extra_headers={"X-Request-Id": request_id},
    )
    
    print("-" * 50)
    print("✓ Prefill 完成")
    print(f"Prefill 返回: {prefill_response.choices[0].text}")
    
    # 3: Decode 阶段
    # 从 prefill 响应中获取 KV cache 参数，传递给 decode 实例
    decode_response = decode_client.completions.create(
        model=model,
        prompt="This prompt is ignored during decode",  # Decode 不处理原始 prompt
        max_tokens=150,  # 生成最多 150 个 token
        extra_body={
            "kv_transfer_params": prefill_response.kv_transfer_params  # 传递 KV cache 信息
        },
        extra_headers={"X-Request-Id": request_id},  # 保持同一请求 ID
    )
    
    print("-" * 50)
    print("✓ Decode 完成")
    print(f"最终返回: {decode_response.choices[0].text}")

except Exception as e:
    print(f"❌ 分离式服务出错: {e}")
    print("请检查 prefill 和 decode 实例均已启动且可访问")
```
