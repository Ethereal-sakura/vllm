# 分离式预填充（实验性）

本页将为你介绍 vLLM 中的分离式预填充（disaggregated prefilling）功能。

!!! note
    此功能目前为实验性，未来可能会有变动。

## 为什么要使用分离式预填充？

主要有两个原因：

- **分别优化首次生成令牌时间（TTFT）和令牌间延迟（ITL）**。分离式预填充将大语言模型（LLM）推理中的预填充和解码阶段安排在不同的 vLLM 实例中。这样，你可以灵活地为不同阶段分配不同的并行策略（比如 `tp` 和 `pp`），从而调优 TTFT 而不影响 ITL，或调优 ITL 而不影响 TTFT。
- **控制尾部 ITL**。如果不使用分离式预填充，vLLM 可能会在一个请求的解码过程中插入一些预填充任务，导致尾部延迟升高。分离式预填充可以帮助你解决这个问题，更好地控制尾部 ITL。虽然通过设置合适的分块预填充（chunked prefill）大小也能达到类似效果，但实际应用中很难确定最佳分块参数。因此，分离式预填充是一种更可靠的控制尾部 ITL 的方法。

!!! note
    分离式预填充不会提升整体吞吐量。

## 使用示例

详见 [examples/online_serving/disaggregated_prefill.sh](../../examples/online_serving/disaggregated_prefill.sh)，这里提供了分离式预填充的使用示例。

目前支持以下 5 种连接器（connector）类型：

- **SharedStorageConnector**：参考 [examples/offline_inference/disaggregated-prefill-v1/run.sh](../../examples/offline_inference/disaggregated-prefill-v1/run.sh) 获取 SharedStorageConnector 分离式预填充的用法。
- **LMCacheConnectorV1**：参考 [examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh) 获取 LMCacheConnectorV1 分离式预填充的用法，底层使用 NIXL 进行 KV 传输。
- **NixlConnector**：参考 [tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh) 获取 NixlConnector 分离式预填充的用法，支持完全异步的发送/接收。详细使用说明见 [NixlConnector 使用指南](nixl_connector_usage.md)。
- **P2pNcclConnector**：参考 [examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh](../../examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh) 获取 P2pNcclConnector 分离式预填充的用法。
- **MultiConnector**：利用 KVTransferConfig 中已存在的 kv_connector_extra_config: dict[str, Any]，可以有序地存放你需要的多个连接器参数。例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}]}}'
  ```

对于 NixlConnector，你还可以指定一个或多个 NIXL_Backend，例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'
  ```

- **OffloadingConnector**：支持将 KV 数据卸载到 CPU 内存，并可自定义 CPU 块大小（以 token 为单位）及每个 worker 分配的块数量：

  ```bash
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size": 64, "num_cpu_blocks": 1000}}'
  ```

## 基准测试

分离式预填充相关的性能测试请参考 [benchmarks/disagg_benchmarks](../../benchmarks/disagg_benchmarks)。

## 开发说明

分离式预填充的实现方式是同时运行两个 vLLM 实例：一个用于预填充（称为 prefill 实例），一个用于解码（称为 decode 实例），然后利用连接器将 KV 缓存和结果从预填充实例传递给解码实例。

所有分离式预填充的实现均位于 `vllm/distributed/kv_transfer` 目录下。

分离式预填充的核心抽象：

- **Connector（连接器）**：Connector 允许 **kv consumer** 从 **kv producer** 获取一批请求的 KV 缓存。
- **LookupBuffer**：LookupBuffer 提供两个接口：`insert` 插入 KV 缓存，以及 `drop_select` 删除并选取 KV 缓存。`insert` 和 `drop_select` 的语义类似 SQL，其中 `insert` 是插入 KV 缓存到缓冲区；`drop_select` 是根据条件返回并删除匹配的 KV 缓存。
- **Pipe**：单向 FIFO 管道，用于张量传输。支持 `send_tensor` 和 `recv_tensor`。

!!! note
    `insert` 操作是非阻塞的，`drop_select` 操作是阻塞的。

下图展示了上述三个抽象的组织方式：

![分离式预填充核心抽象](../assets/features/disagg_prefill/abstraction.jpg)

分离式预填充的工作流程如下：

![分离式预填充流程图](../assets/features/disagg_prefill/overview.jpg)

图中的 `buffer` 对应于 LookupBuffer 的 `insert` API，`drop_select` 对应于 LookupBuffer 的 `drop_select` API。

现在，vLLM 中的每个进程都有自己的连接器。具体分为：

- Scheduler connector：与调度器进程同处于一个进程中，负责调度 KV 缓存传输操作的连接器。
- Worker connectors：分布在各个 worker 进程中的连接器，负责实际执行 KV 缓存传输操作。

下图展示了这两类连接器的组织结构：

![分离式预填充高层设计](../assets/features/disagg_prefill/high_level_design.png)

下图展示了 worker connector 如何与注意力模块协作，实现分层 KV 缓存的存储与加载：

![分离式预填充流程图](../assets/features/disagg_prefill/workflow.png)

## 第三方贡献

分离式预填充高度依赖底层基础设施，因此 vLLM 在生产环境下主要依赖第三方连接器实现分离式预填充（同时 vLLM 团队会积极审核和合并第三方连接器相关的 PR）。

推荐三种实现方式：

- **完全自定义连接器**：实现你自己的 `Connector`，并调用第三方库来发送和接收 KV 缓存，甚至包括自定义预填充相关的模型输入。这种方式灵活性最高，但未来 vLLM 版本升级时可能会有兼容性风险。
- **类数据库连接器**：自定义实现 `LookupBuffer`，并支持类似 SQL 的 `insert` 和 `drop_select` 接口。
- **分布式点对点连接器**：自定义实现 `Pipe`，支持类似于 `torch.distributed` 的 `send_tensor` 和 `recv_tensor` 接口。