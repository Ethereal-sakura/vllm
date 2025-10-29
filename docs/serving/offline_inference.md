# 离线推理

你可以在自己的代码中使用 vLLM 的 [`LLM`][vllm.LLM] 类进行离线推理。

比如，下面的代码会从 HuggingFace 下载 [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) 模型，
并用默认配置在 vLLM 中运行该模型。

```python
from vllm import LLM

# 初始化 vLLM 引擎
llm = LLM(model="facebook/opt-125m")
```

初始化 `LLM` 实例后，可以通过相关 API 执行模型推理。
具体可用的 API 取决于模型类型：

- [生成式模型](../models/generative_models.md) 会输出 logprobs，最终的文本结果由这些概率采样得到。
- [池化模型](../models/pooling_models.md) 会直接输出它们的隐藏状态。

!!! info
    [API 参考](../api/README.md#offline-inference)

## Ray Data LLM API

Ray Data LLM 是一种替代的离线推理 API，底层同样使用 vLLM 引擎。
这一 API 提供了多项开箱即用的功能，能够简化大规模、高效利用 GPU 的推理流程：

- 流式执行可处理超出集群总内存的数据集。
- 自动分片、负载均衡与自动扩容，工作任务分布在 Ray 集群中，支持容错。
- 持续批处理让 vLLM 副本始终保持高效，最大化 GPU 利用率。
- 支持张量并行和流水线并行，轻松实现多 GPU 高效推理。
- 支持主流文件格式和云对象存储的读写操作。
- 无需修改代码即可扩展工作负载规模。

??? code

    ```python
    import ray  # 需要 ray>=2.44.1
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
    processor = build_llm_processor(
        config,
        preprocess=lambda row: {
            "messages": [
                {"role": "system", "content": "你是一个用诗句补全未完成俳句的机器人。"},
                {"role": "user", "content": row["item"]},
            ],
            "sampling_params": {"temperature": 0.3, "max_tokens": 250},
        },
        postprocess=lambda row: {"answer": row["generated_text"]},
    )

    ds = ray.data.from_items(["An old silent pond..."])
    ds = processor(ds)
    ds.write_parquet("local:///tmp/data/")
    ```

关于 Ray Data LLM API 的更多信息，可参考 [Ray Data LLM 文档](https://docs.ray.io/en/latest/data/working-with-llms.html) 
