# 指标（Metrics）

vLLM 提供了丰富的指标体系，支持 V1 引擎的可观测性和容量规划。

## 目标

- 覆盖引擎级和请求级的指标，便于生产环境监控。
- 优先集成 Prometheus，因为我们预期生产环境主要会用它。
- 支持日志输出（比如将指标打印到 info 日志），方便临时测试、调试、开发和探索性使用场景。

## 背景

vLLM 中的指标主要分为以下几类：

1. 服务器级指标：用于跟踪 LLM 引擎全局状态和性能，通常通过 Prometheus 以 Gauge（仪表盘）或 Counter（计数器）形式暴露。
2. 请求级指标：用于跟踪单个请求的特征（如请求大小和处理时长），通常以 Histogram（直方图）形式暴露，也是 SRE 监控 vLLM 时关注的服务水平目标（SLO）。

简而言之，服务器级指标用于解释请求级指标的变化原因。

### 指标总览

### v1 指标

在 v1 版本中，以下指标通过 Prometheus 兼容的 `/metrics` 端点暴露，统一使用 `vllm:` 前缀：

- `vllm:num_requests_running` (Gauge) - 当前正在运行的请求数
- `vllm:num_requests_waiting` (Gauge) - 当前等待中的请求数
- `vllm:kv_cache_usage_perc` (Gauge) - KV 缓存块的使用比例（0–1）
- `vllm:prefix_cache_queries` (Counter) - 前缀缓存查询次数
- `vllm:prefix_cache_hits` (Counter) - 前缀缓存命中次数
- `vllm:mm_cache_queries` (Counter) - （多模态模型）多模态缓存查询次数
- `vllm:mm_cache_hits` (Counter) - （多模态模型）多模态缓存命中次数
- `vllm:num_preemptions_total` (Counter) - 抢占次数
- `vllm:prompt_tokens_total` (Counter) - 处理过的提示词 token 总数
- `vllm:generation_tokens_total` (Counter) - 生成的 token 总数
- `vllm:iteration_tokens_total` (Histogram) - 每次引擎迭代处理的 token 数量分布
- `vllm:cache_config_info` (Gauge) - 缓存配置信息
- `vllm:request_success_total` (Counter) - 已完成请求的数量（按完成原因分类）
- `vllm:request_prompt_tokens` (Histogram) - 输入提示词 token 数量分布
- `vllm:request_generation_tokens` (Histogram) - 生成 token 数量分布
- `vllm:request_params_n` (Histogram) - 请求参数 n 的分布
- `vllm:request_params_max_tokens` (Histogram) - 请求参数 max_tokens 的分布
- `vllm:time_to_first_token_seconds` (Histogram) - 首 token 时间（TTFT）
- `vllm:inter_token_latency_seconds` (Histogram) - token 间延迟
- `vllm:e2e_request_latency_seconds` (Histogram) - 请求端到端延迟
- `vllm:request_queue_time_seconds` (Histogram) - 排队等待时长
- `vllm:request_inference_time_seconds` (Histogram) - 推理时长
- `vllm:request_prefill_time_seconds` (Histogram) - 预填充时长
- `vllm:request_decode_time_seconds` (Histogram) - 解码时长

详细文档参见 [推理与服务 -> 生产环境指标](../usage/metrics.md)。

### Grafana 仪表盘

vLLM 还提供了[参考示例](../../examples/online_serving/prometheus_grafana/README.md)，展示如何借助 Prometheus 收集、存储这些指标，并用 Grafana 仪表盘进行可视化。

Grafana 仪表盘暴露的指标子集，体现了哪些指标尤其关键：

- `vllm:e2e_request_latency_seconds_bucket` - 端到端请求延迟（单位：秒）
- `vllm:prompt_tokens_total` - 提示词 token 总数
- `vllm:generation_tokens_total` - 生成 token 总数
- `vllm:time_per_output_token_seconds` - token 间延迟（TPOT，单位：秒）
- `vllm:time_to_first_token_seconds` - 首 token 时间（TTFT，单位：秒）
- `vllm:num_requests_running`（以及 `_swapped` 和 `_waiting`）- 各状态（RUNNING、WAITING、SWAPPED）下的请求数
- `vllm:gpu_cache_usage_perc` - GPU 缓存块使用率
- `vllm:request_prompt_tokens` - 请求提示词长度
- `vllm:request_generation_tokens` - 请求生成长度
- `vllm:request_success_total` - 按完成原因统计的请求数（如生成 EOS token 或达到最大序列长度）
- `vllm:request_queue_time_seconds` - 排队时间
- `vllm:request_prefill_time_seconds` - 预填充时长
- `vllm:request_decode_time_seconds` - 解码时长
- `vllm:request_max_num_generation_tokens` - 每个序列组的最大生成 token 数

更多背景和选择原则，参见 [新增仪表盘的 PR](https://github.com/vllm-project/vllm/pull/2316)。

### Prometheus 客户端库

最初我们[采用 aioprometheus 库集成 Prometheus](https://github.com/vllm-project/vllm/pull/1890)，但很快切换到了 [prometheus_client](https://github.com/vllm-project/vllm/pull/2730)。详细原因可见相关 PR。

迁移过程中，一度丢失了用于追踪 HTTP 指标的 `MetricsMiddleware`，但后来通过 [prometheus_fastapi_instrumentator](https://github.com/vllm-project/vllm/pull/15657) 恢复：

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### 多进程模式

历史上，指标收集是在引擎核心进程完成，通过多进程模式让 API 服务器进程能获取这些数据。参考 <https://github.com/vllm-project/vllm/pull/7279>

近期，指标改为在 API 服务器进程收集，只有当 `--api-server-count > 1` 时才使用多进程。详情可见 <https://github.com/vllm-project/vllm/pull/17546> 及 [API 服务器扩展部署说明](../serving/data_parallel_deployment.md#internal-load-balancing)。

### Python/进程内置指标

`prometheus_client` 默认支持以下指标，但在启用多进程模式时不会暴露：

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

因此，当 `--api-server-count > 1` 时，这些指标不可用。其实这些指标也不一定很有意义，因为不能汇总 vLLM 实例的所有进程信息。

## 指标设计

关于指标设计，很多讨论和规划见于 ["更好的可观测性" 议题](https://github.com/vllm-project/vllm/issues/3616)。例如，[详细的规划路线图](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781)。

### 历史 PR

想了解指标设计的由来，可以查看以下相关 PR（添加了原始/已废弃指标）：

- <https://github.com/vllm-project/vllm/pull/1890>
- <https://github.com/vllm-project/vllm/pull/2316>
- <https://github.com/vllm-project/vllm/pull/2730>
- <https://github.com/vllm-project/vllm/pull/4464>
- <https://github.com/vllm-project/vllm/pull/7279>

### 指标实现相关 PR

更多实现细节，参考以下 PR 和议题 <https://github.com/vllm-project/vllm/issues/10582>：

- <https://github.com/vllm-project/vllm/pull/11962>
- <https://github.com/vllm-project/vllm/pull/11973>
- <https://github.com/vllm-project/vllm/pull/10907>
- <https://github.com/vllm-project/vllm/pull/12416>
- <https://github.com/vllm-project/vllm/pull/12478>
- <https://github.com/vllm-project/vllm/pull/12516>
- <https://github.com/vllm-project/vllm/pull/12530>
- <https://github.com/vllm-project/vllm/pull/12561>
- <https://github.com/vllm-project/vllm/pull/12579>
- <https://github.com/vllm-project/vllm/pull/12592>
- <https://github.com/vllm-project/vllm/pull/12644>

### 指标收集

在 v1 设计中，我们希望将计算与开销移出引擎核心进程，从而减少每次前向推理之间的延迟。

V1 EngineCore 的设计理念：

- EngineCore 是核心循环，对性能要求最高
- AsyncLLM 是外层循环，能与 GPU 执行并行（理想情况下），因此各类“开销”应尽量放在这里。`AsyncLLM.output_handler_loop` 是指标统计的理想位置。

实现方式是：在前端 API 服务器收集指标，基于引擎核心进程返回的 `EngineCoreOutputs` 信息进行统计。

### 时长间隔计算

许多指标关注的是请求处理过程中各事件的时间间隔。最佳实践是用“单调时间”（`time.monotonic()`）而不是“系统时间”（`time.time()`）计算间隔，因为单调时间不受系统时钟变化影响（比如 NTP 校准）。

但要注意，单调时钟是进程私有的——不同进程的单调时间基准点不一样。所以不能比较不同进程的单调时间戳。

因此，计算间隔时必须比较来自同一进程的两个单调时间戳。

### 调度器统计信息

引擎核心进程会收集一些调度器相关的关键统计数据，例如上一次调度后仍在等待或已被调度的请求数，并将这些数据包含在 `EngineCoreOutputs` 中。

### 引擎核心事件

引擎核心还会记录每个请求一些关键事件的时间戳，供前端计算事件间的时间间隔。

这些事件包括：

- `QUEUED` - 请求被引擎核心接收并加入调度队列的时刻
- `SCHEDULED` - 请求首次被调度执行的时刻
- `PREEMPTED` - 请求被放回等待队列（为其他请求腾空间），未来会重新调度并重新进入 prefill 阶段
- `NEW_TOKENS` - `EngineCoreOutput` 输出中的新 token 被生成的时刻。对同一轮迭代的所有请求，用同一个时间戳记录。

间隔计算如下：

- 排队间隔：`QUEUED` 到最近一次 `SCHEDULED` 的时间
- 预填充间隔：最近一次 `SCHEDULED` 到首次 `NEW_TOKENS`
- 解码间隔：首次（在最近一次 `SCHEDULED` 后）到最后一次 `NEW_TOKENS`
- 推理间隔：最近一次 `SCHEDULED` 到最后一次 `NEW_TOKENS`
- Token 间隔：连续两次 `NEW_TOKENS` 之间的时间

换句话说：

![常规间隔计算示意图](../assets/design/metrics/intervals-1.png)

我们曾尝试让前端自行计算这些间隔，但前端无法获知 `QUEUED` 和 `SCHEDULED` 的具体时间。而且要保证间隔计算用的是同一进程的单调时间戳，所以必须由引擎核心为所有相关事件记录时间戳。

#### 间隔计算与抢占（Preemption）

如果在解码阶段发生抢占（Preemption），已生成的 token 会被复用，抢占影响的是 token 间隔、解码间隔和推理间隔。

![解码期间抢占的间隔计算](../assets/design/metrics/intervals-2.png)

如果在预填充阶段发生抢占（假设这种情况可能出现），抢占会影响首 token 时间和预填充间隔。

![预填充期间抢占的间隔计算](../assets/design/metrics/intervals-3.png)

### 前端统计收集

每处理一次 `EngineCoreOutputs`（即引擎核心一次迭代的输出），前端会收集以下统计信息：

- 本次迭代中新生成的 token 总数
- 本次迭代完成预填充的请求所处理的提示词 token 总数
- 本次迭代被调度请求的排队时间间隔
- 本次迭代完成预填充请求的预填充间隔
- 所有包含在本次迭代中的请求的 token 间隔（TPOT）
- 本次迭代完成预填充的请求的 TTFT（首 token 时间），此间隔是相对于请求首次到达前端（`arrival_time`）计算的，以考虑输入处理时间

对于本次迭代完成的请求，还会记录：

- 推理和解码间隔（如前文所述，与调度和首 token 事件相关）
- 端到端延迟（前端 `arrival_time` 到收到最后一个 token 的时间）

### 指标发布 - 日志

`LoggingStatLogger` 指标发布器每 5 秒输出一次 INFO 日志，内容包括：

- 当前运行/等待中的请求数
- 当前 GPU 缓存使用率
- 过去 5 秒处理的提示词 token 数（吞吐率）
- 过去 5 秒生成的新 token 数（生成速率）
- 最近 1000 次 kv-cache 查询的前缀缓存命中率

### 指标发布 - Prometheus

`PrometheusStatLogger` 指标发布器通过 `/metrics` HTTP 端点提供 Prometheus 兼容格式的指标。Prometheus 实例可定期（例如每秒）拉取数据，存储到自身的时序数据库。通常结合 Grafana 实现可视化。

Prometheus 支持以下指标类型：

- Counter（计数器）：只增不减，通常在 vLLM 实例重启时归零。例如，实例运行期间生成的 token 总数。
- Gauge（仪表盘）：数值可增可减，比如当前调度中的请求数。
- Histogram（直方图）：按区间统计样本数量。例如，TTFT 在 <1ms、<5ms、<10ms、<20ms 等区间内的请求数量。

Prometheus 的指标还可以打标签，方便按标签聚合。vLLM 每条指标都会加上 `model_name` 标签，用于标识具体服务的模型。

示例输出：

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3