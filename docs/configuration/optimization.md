# 优化与性能调优

本指南介绍 vLLM V1 的优化策略和性能调优方法。

!!! tip
    内存不足？请查阅[此指南](./conserving_memory.md)，了解如何节省内存资源。

## 抢占机制（Preemption）

由于 Transformer 架构的自回归特性，KV 缓存空间有时无法满足所有批量请求的需求。
这时，vLLM 会通过抢占部分请求来释放 KV 缓存空间，供其他请求使用。被抢占的请求会在 KV 缓存空间充足时重新计算。
如果出现这种情况，您可能会看到如下警告：

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

这种机制保障了系统的稳定性，但抢占与重新计算可能会增加整体延迟。
如果您经常遇到抢占现象，可以考虑以下措施：

- 增加 `gpu_memory_utilization`。vLLM 会按此比例预分配 GPU 缓存，提高利用率可扩展 KV 缓存空间。
- 降低 `max_num_seqs` 或 `max_num_batched_tokens`。减少每批并发请求数量，从而降低 KV 缓存需求。
- 增加 `tensor_parallel_size`。模型参数会在多块 GPU 上分片，让每块 GPU 有更多空间用于 KV 缓存。但并行度过高可能带来同步开销。
- 增加 `pipeline_parallel_size`。模型层会在多块 GPU 间分布，减少每块 GPU 对模型权重的内存消耗，间接释放更多 KV 缓存空间。但并行度过高也可能影响延迟。

您可以通过 vLLM 暴露的 Prometheus 指标监控抢占请求数量，同时设置 `disable_log_stats=False` 可记录累计抢占次数。

在 vLLM V1 中，默认的抢占模式为 `RECOMPUTE`，而非 `SWAP`，因为在 V1 架构下重新计算的开销更低。

## 分块预填（Chunked Prefill）

分块预填让 vLLM 能将大型预填请求拆分为更小的块，并与解码请求一起批量处理。
该特性通过均衡计算密集型（预填）与内存密集型（解码）操作，提升吞吐和延迟表现。

在 vLLM V1 中，**分块预填始终默认开启**。这与 vLLM V0 不同，V0 是根据模型特性条件开启。

启用分块预填后，调度策略会优先处理解码请求。系统会先将所有待处理的解码请求合批，再安排预填操作。
如果 `max_num_batched_tokens` 预算有剩余，则会调度待处理的预填任务。如果单个预填请求超出 `max_num_batched_tokens`，则会自动将其分块。

这种调度方式有两个优势：

- 优先处理解码请求，有助于提升 ITL（单 token 延迟）与生成速度。
- 计算密集型与内存密集型请求可同时进入同一批次，提高 GPU 利用率。

### 分块预填的性能调优

您可以通过调整 `max_num_batched_tokens` 参数来优化性能：

- 较小的数值（如 2048）有助于降低 ITL，因为预填请求更少，不会拖慢解码流程。
- 较大的数值可提升首 token 响应速度（TTFT），一次批量可处理更多预填 token。
- 若追求高吞吐，建议将 `max_num_batched_tokens` 设置为 8192 以上，尤其在大 GPU 跑小模型时。
- 如果 `max_num_batched_tokens` 与 `max_model_len` 相同，则调度策略几乎等同于 V0 的默认方式（但仍优先解码）。

```python
from vllm import LLM

# 通过设置 max_num_batched_tokens 调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

更多细节可参考相关论文（<https://arxiv.org/pdf/2401.08671> 或 <https://arxiv.org/pdf/2308.16369>）

## 并行策略

vLLM 支持多种并行机制，可自由组合，以适配不同硬件环境下的性能优化需求。

### 张量并行（Tensor Parallelism, TP）

张量并行会在每一层模型内，将参数分片到多块 GPU 上。这是大模型单节点推理最常用的方式。

**适用场景：**

- 模型体积过大，无法放入单块 GPU
- 需降低单块 GPU 的内存压力，为更高吞吐留出更多 KV 缓存空间

```python
from vllm import LLM

# 将模型拆分到 4 块 GPU 上
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

对于参数量极大的模型（如 70B），张量并行是必需的。

### 流水线并行（Pipeline Parallelism, PP）

流水线并行会将模型各层分布到多块 GPU 上，每块 GPU 依次处理模型的不同部分。

**适用场景：**

- 已充分使用张量并行，但还需进一步分布模型，或跨节点部署
- 模型结构较深且较窄，按层分布比张量分片更高效

流水线并行可与张量并行结合，用于超大模型：

```python
from vllm import LLM

# 流水线并行与张量并行组合
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)
```

### 专家并行（Expert Parallelism, EP）

专家并行专为专家混合（Mixture of Experts, MoE）模型设计，将不同专家网络分布到多块 GPU 上。

**适用场景：**

- 专门针对 MoE 模型（如 DeepSeekV3、Qwen3MoE、Llama-4 等）
- 需在多块 GPU 间均衡专家计算负载

设置 `enable_expert_parallel=True` 即可开启专家并行，对 MoE 层采用专家并行而非张量并行。
专家并行的度数与张量并行设置一致。

### 数据并行（Data Parallelism, DP）

数据并行会将整个模型复制到多组 GPU 上，并行处理不同批次的请求。

**适用场景：**

- 有足够 GPU，可复制完整模型
- 需提升吞吐量而非扩大模型规模
- 多用户环境下，批次间隔离更有益

数据并行可与其他并行机制结合，通过设置 `data_parallel_size=N` 启用。
注意：MoE 层会按照张量并行数与数据并行数的乘积进行分片。

### 多模态编码器的批级 DP

默认情况下，多模态编码器的权重分片方式与语言解码器一致，采用 TP，
以降低每块 GPU 的内存和计算负载。

但由于多模态编码器远小于语言解码器，TP 带来的收益有限，
且每层后都需执行 all-reduce，通信开销较大。

因此，可以改为用 TP 对输入数据进行分片，即批级 DP。实验证明，
当 `tensor_parallel_size=8` 时，这种方式可提升约 10% 的吞吐和 TTFT。
对于采用未针对硬件优化的 Conv3D 操作的视觉编码器，批级 DP 可再提升 40%。

需要注意的是，多模态编码器的权重会在每个 TP rank 上复制，
因此会略微增加内存消耗，若模型本身刚好能装下，可能导致 OOM。

通过设置 `mm_encoder_tp_mode="data"` 即可启用批级 DP，例如：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size=4,
    # 当 mm_encoder_tp_mode="data" 时，
    # 视觉编码器将数据分片，TP=4 实际为 DP=4，
    # 但这不影响语言解码器在专家并行下的 DP 设置。
    mm_encoder_tp_mode="data",
    # 语言解码器始终采用 TP=4 分片权重，与 mm_encoder_tp_mode 设置无关
)
```

!!! important
    批级 DP 不等同于 API 请求级 DP
    （后者通过 `data_parallel_size` 控制）。

批级 DP 需针对具体模型实现，并在模型类中设置 `supports_encoder_tp_data = True`。
但无论如何，使用该特性时需在引擎参数中设置 `mm_encoder_tp_mode="data"`。

已知支持的模型（附基准测试链接）：

- dots_ocr (<https://github.com/vllm-project/vllm/pull/25466>)
- GLM-4.1V 及以上版本 (<https://github.com/vllm-project/vllm/pull/23168>)
- InternVL (<https://github.com/vllm-project/vllm/pull/23909>)
- Kimi-VL (<https://github.com/vllm-project/vllm/pull/23817>)
- Llama4 (<https://github.com/vllm-project/vllm/pull/18368>)
- MiniCPM-V-2.5 及以上版本 (<https://github.com/vllm-project/vllm/pull/23327>, <https://github.com/vllm-project/vllm/pull/23948>)
- Qwen2-VL 及以上版本 (<https://github.com/vllm-project/vllm/pull/22742>, <https://github.com/vllm-project/vllm/pull/24955>, <https://github.com/vllm-project/vllm/pull/25445>)
- Step3 (<https://github.com/vllm-project/vllm/pull/22697>)

## 输入处理

### 并行处理

可以通过 [API 服务端扩展](../serving/data_parallel_deployment.md#internal-load-balancing) 实现输入处理的并行化。
当输入处理（在 API 服务端执行）成为瓶颈，而模型执行（在引擎核心）不是瓶颈，且 CPU 资源充足时，这很有用。

```console
# 启动 4 个 API 服务端进程和 1 个引擎核心进程
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4

# 启动 4 个 API 服务端进程和 2 个引擎核心进程
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4 -dp 2
```

!!! note
    API 服务端扩展仅适用于在线推理场景。

!!! warning
    默认情况下，每个 API 服务端会用 8 个 CPU 线程从请求数据中加载媒体项（如图片）。

    若启用 API 服务端扩展，请合理调整 `VLLM_MEDIA_LOADING_THREAD_COUNT`，避免耗尽 CPU 资源。

!!! note
    API 服务端扩展会禁用[多模态 IPC 缓存](#ipc-caching)，
    因为该缓存要求 API 与引擎核心进程一一对应。

    这不会影响[多模态处理器缓存](#processor-caching)。

## 多模态缓存

多模态缓存用于避免重复传输和处理同一多模态数据，在多轮对话场景下尤为常见。

### 处理器缓存

多模态处理器缓存会自动开启，
避免在 `BaseMultiModalProcessor` 中对同一多模态输入重复处理。

### IPC 缓存

当 API（`P0`）与引擎核心（`P1`）进程一一对应时，
多模态 IPC 缓存会自动启用，
避免在两者间重复传输同一多模态输入。

#### 键复制缓存（Key-Replicated Cache）

默认情况下，IPC 缓存采用**键复制缓存**，即缓存键同时存在于 API（`P0`）和引擎核心（`P1`）进程，
但实际缓存数据仅存放在 `P1`。

#### 共享内存缓存（Shared Memory Cache）

如涉及多个工作进程（例如 TP > 1），**共享内存缓存**更高效。
通过设置 `mm_processor_cache_type="shm"` 可启用。
此模式下，缓存键存储在 `P0`，而缓存数据存放在所有进程可访问的共享内存中。

### 配置方式

可通过设置 `mm_processor_cache_gb`（默认 4 GiB）调整缓存大小。

如果缓存效果不明显，也可通过 `mm_processor_cache_gb=0` 完全关闭 IPC 与处理器缓存。

示例：

```python
# 使用更大缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=8,
)

# 启用共享内存 IPC 缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    tensor_parallel_size=2,
    mm_processor_cache_type="shm",
    mm_processor_cache_gb=8,
)

# 禁用缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=0,
)
```

### 缓存分布

根据配置，`P0` 和 `P1` 上的多模态缓存内容如下：

| mm_processor_cache_type | 缓存类型 | `P0` 缓存 | `P1` 引擎缓存 | `P1` 工作缓存 | 最大内存占用 |
|-------------------|-------------|------------|------------|-------------|-------------|
| lru | 处理器缓存 | K + V | N/A | N/A | `mm_processor_cache_gb * data_parallel_size` |
| lru | 键复制缓存 | K | K + V | N/A | `mm_processor_cache_gb * api_server_count` |
| shm | 共享内存缓存 | K | N/A | V | `mm_processor_cache_gb * api_server_count` |
| N/A | 禁用 | N/A | N/A | N/A | `0` |

K：存储多模态数据的哈希值  
V：存储多模态数据处理后的张量数据