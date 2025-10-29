# 池化模型

vLLM 还支持池化模型，包括嵌入（embedding）、分类（classification）和奖励（reward）模型。

在 vLLM 中，池化模型实现了 [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] 接口。这些模型会通过 [Pooler][vllm.model_executor.layers.pooler.Pooler] 层在返回结果前提取输入的最终隐藏状态。

!!! note
    目前我们主要是为了使用方便而支持池化模型。相比直接使用 HF Transformers 或 Sentence Transformers，性能并不一定有提升。

    我们正在计划优化 vLLM 的池化模型。如果你有建议，欢迎在 <https://github.com/vllm-project/vllm/issues/21796> 留言！

## 配置说明

### 模型运行模式

通过参数 `--runner pooling` 可以将模型以池化模式运行。

!!! tip
    大多数情况下无需手动设置该参数，vLLM 会自动通过 `--runner auto` 检测并选择合适的模型运行模式。

### 模型转换

vLLM 可以通过 `--convert <type>` 参数自动适配不同池化任务的模型。

如果设置了 `--runner pooling`（手动或自动）但模型未实现 [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] 接口，vLLM 会根据下表中的架构名称自动转换模型：

| 架构名称                                       | `--convert` | 支持的池化任务                   |
|-----------------------------------------------|-------------|----------------------------------|
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model` | `embed`     | `encode`, `embed`                |
| `*For*Classification`, `*ClassificationModel`   | `classify`  | `encode`, `classify`, `score`    |
| `*ForRewardModeling`, `*RewardModel`            | `reward`    | `encode`                         |

!!! tip
    你可以通过显式设置 `--convert <type>` 参数来指定模型转换类型。

### 池化任务

每个 vLLM 池化模型根据 [Pooler.get_supported_tasks][vllm.model_executor.layers.pooler.Pooler.get_supported_tasks] 支持一个或多个任务，并开放相应的 API：

| 任务         | 可用 API                                      |
|--------------|-----------------------------------------------|
| `encode`     | `LLM.reward(...)`                             |
| `embed`      | `LLM.embed(...)`, `LLM.score(...)`\*          |
| `classify`   | `LLM.classify(...)`                           |
| `score`      | `LLM.score(...)`                              |

\* 如果模型不支持 `score` 任务，`LLM.score(...)` 会回退使用 `embed` 任务。

### Pooler 配置

#### 预定义模型

如果模型定义的 [Pooler][vllm.model_executor.layers.pooler.Pooler] 支持 `pooler_config`，可以通过 `--pooler-config` 参数覆盖部分属性。

#### 已转换模型

如果通过 `--convert` 转换了模型（见上文），分配给各任务的 pooler 默认属性如下：

| 任务         | 池化类型   | 归一化   | Softmax |
|--------------|------------|----------|---------|
| `reward`     | `ALL`      | ❌       | ❌      |
| `embed`      | `LAST`     | ✅︎       | ❌      |
| `classify`   | `LAST`     | ❌       | ✅︎     |

当加载 [Sentence Transformers](https://huggingface.co/sentence-transformers) 模型时，模型的 Sentence Transformers 配置文件（`modules.json`）优先于默认设置。

你还可以通过 `--pooler-config` 进一步自定义设置，优先级高于模型和 Sentence Transformers 的默认配置。

## 离线推理

[LLM][vllm.LLM] 类提供了多种离线推理方法。关于模型初始化的参数选项，请参考 [配置文档](../api/README.md#configuration)。

### `LLM.embed`

[embed][vllm.LLM.embed] 方法会为每个 prompt 输出一个嵌入向量。主要用于嵌入模型。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

更多代码示例见：[examples/offline_inference/basic/embed.py](../../examples/offline_inference/basic/embed.py)

### `LLM.classify`

[classify][vllm.LLM.classify] 方法会为每个 prompt 输出一个概率向量。主要用于分类模型。

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

代码示例见：[examples/offline_inference/basic/classify.py](../../examples/offline_inference/basic/classify.py)

### `LLM.score`

[score][vllm.LLM.score] 方法用于输出句子对之间的相似度分数。适用于嵌入模型和 cross-encoder（交叉编码器）模型。嵌入模型采用余弦相似度，而 [cross-encoder 模型](https://www.sbert.net/examples/applications/cross-encoder/README.html) 常用于 RAG 系统中对候选查询-文档对进行重排序。

!!! note
    vLLM 仅负责 RAG 的模型推理部分（如嵌入、重排序）。如需更高层的 RAG 流程，建议使用 [LangChain](https://github.com/langchain-ai/langchain) 等集成框架。

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

代码示例见：[examples/offline_inference/basic/score.py](../../examples/offline_inference/basic/score.py)

### `LLM.reward`

[reward][vllm.LLM.reward] 方法适用于全部的奖励模型。它会直接返回提取的隐藏状态。

```python
from vllm import LLM

llm = LLM(model="internlm/internlm2-1_8b-reward", runner="pooling", trust_remote_code=True)
(output,) = llm.reward("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

代码示例见：[examples/offline_inference/basic/reward.py](../../examples/offline_inference/basic/reward.py)

### `LLM.encode`

[encode][vllm.LLM.encode] 方法适用于所有池化模型，直接返回提取的隐藏状态。

!!! note
    使用 `LLM.encode` 时，建议使用更具体的方法或直接指定任务类型：

    - 嵌入任务请用 `LLM.embed(...)` 或设置 `pooling_task="embed"`
    - 分类 logits 请用 `LLM.classify(...)` 或设置 `pooling_task="classify"`
    - 奖励任务请用 `LLM.reward(...)` 或设置 `pooling_task="reward"`
    - 相似度分数请用 `LLM.score(...)`

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## 在线服务

我们的 [OpenAI 兼容服务端](../serving/openai_compatible_server.md) 提供了与离线 API 对应的接口：

- [池化 API](../serving/openai_compatible_server.md#pooling-api) 类似于 `LLM.encode`，适用于所有池化模型类型
- [嵌入 API](../serving/openai_compatible_server.md#embeddings-api) 类似于 `LLM.embed`，支持文本和[多模态输入](../features/multimodal_inputs.md)
- [分类 API](../serving/openai_compatible_server.md#classification-api) 对应 `LLM.classify`，适用于序列分类模型
- [分数 API](../serving/openai_compatible_server.md#score-api) 对应于交叉编码器模型的 `LLM.score`

## Matryoshka 嵌入

[Matryoshka Embeddings（套娃嵌入）](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) 或 [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) 是一种嵌入模型训练技术，允许用户在性能和成本之间灵活权衡。

!!! warning
    并不是所有嵌入模型都采用了 Matryoshka 表示学习。为避免误用 `dimensions` 参数，vLLM 如果检测到模型不支持 Matryoshka 嵌入，会直接报错。

    例如，使用 `BAAI/bge-m3` 模型时设置了 `dimensions` 参数，会报如下错误：

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

### 手动启用 Matryoshka 嵌入

目前还没有官方接口用于声明模型支持 Matryoshka 嵌入。在 vLLM 中，如果模型的 `config.json` 中 `is_matryoshka` 设置为 `True`，则允许任意维度的输出。你可以用 `matryoshka_dimensions` 控制允许的输出维度。

对于支持 Matryoshka 嵌入但 vLLM 未识别的模型，可以手动覆盖配置：离线方式使用 `hf_overrides={"is_matryoshka": True}` 或 `hf_overrides={"matryoshka_dimensions": [<允许的输出维度>]}`，在线方式用 `--hf-overrides '{"is_matryoshka": true}'` 或 `--hf-overrides '{"matryoshka_dimensions": [<允许的输出维度>]}'`。

以下是启用 Matryoshka 嵌入的模型服务示例：

```bash
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf-overrides '{"matryoshka_dimensions":[256]}'
```

### 离线推理

对于支持 Matryoshka 嵌入的模型，可以在 [PoolingParams][vllm.PoolingParams] 参数中设置输出维度。

```python
from vllm import LLM, PoolingParams

llm = LLM(
    model="jinaai/jina-embeddings-v3",
    runner="pooling",
    trust_remote_code=True,
)
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
print(outputs[0].outputs)
```

代码示例见：[examples/offline_inference/pooling/embed_matryoshka_fy.py](../../examples/offline_inference/pooling/embed_matryoshka_fy.py)

### 在线推理

用以下命令启动 vllm 服务：

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

支持 Matryoshka 嵌入的模型可以通过 `dimensions` 参数调整输出维度：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

预期输出如下：

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

OpenAI 客户端示例见：[examples/online_serving/pooling/openai_embedding_matryoshka_fy.py](../../examples/online_serving/pooling/openai_embedding_matryoshka_fy.py)
