# 生产环境指标

vLLM 提供了多项指标，便于监控系统的健康状况。这些指标可以通过 vLLM OpenAI 兼容 API 服务器的 `/metrics` 接口获取。

你可以使用 Python 启动服务器，或通过 [Docker](../deployment/docker.md) 启动：

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

然后通过查询该接口获取服务器的最新指标数据：

??? console "输出示例"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

目前开放的指标如下：

??? code

    ```python
    --8<-- "vllm/engine/metrics.py:metrics-definitions"
    ```

注意：当某个指标在 `X.Y` 版本被弃用时，会在 `X.Y+1` 版本中隐藏，但可以通过 `--show-hidden-metrics-for-version=X.Y` 参数重新显示，最终会在 `X.Y+2` 版本中彻底移除。