# LoRA 适配器

本文将介绍如何在基础模型（base model）上，结合 vLLM 使用 [LoRA 适配器](https://arxiv.org/abs/2106.09685)。

只要是实现了 [SupportsLoRA][vllm.model_executor.models.interfaces.SupportsLoRA] 接口的 vLLM 模型，都可以使用 LoRA 适配器。

适配器可以按需、高效地为每个请求动态加载，几乎不会带来额外开销。首先，我们需要下载适配器并保存在本地：

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
```

然后，实例化基础模型，并添加 `enable_lora=True` 参数：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)
```

现在，我们可以通过 `lora_request` 参数，将 prompt 提交给 `llm.generate` 方法。`LoRARequest` 的第一个参数是用于区分适配器的人类可读名称，第二个参数是适配器的全局唯一 ID，第三个参数是 LoRA 适配器的本地路径。

??? code

    ```python
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )
    ```

更多关于异步引擎和高级配置的用法，可以参考 [examples/offline_inference/multilora_inference.py](../../examples/offline_inference/multilora_inference.py)。

## LoRA 适配器服务部署

你也可以通过 OpenAI 接口兼容的 vLLM 服务端来部署 LoRA 适配模型。在启动服务时，使用 `--lora-modules {name}={path} {name}={path}` 指定每个 LoRA 模块：

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```

!!! note
    commit ID `0dfa347e8877a4d4ed19ee56c140fa518470028c` 可能会随着时间变化。请在自己的环境中查看最新的 commit ID，确保使用正确的路径。

服务启动时可以使用其他 LoRA 配置参数（如 `max_loras`、`max_lora_rank`、`max_cpu_loras` 等），这些参数会影响后续所有请求。当你访问 `/models` 接口时，应该可以看到你的 LoRA 适配器和基础模型一并列出（如果没有安装 `jq`，可以参考 [安装指南](https://jqlang.org/download/)）。

??? console "命令示例"

    ```bash
    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-2-7b-hf",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }
    ```

你可以像请求其他模型一样，通过 `model` 参数指定 LoRA 适配器。请求会按照服务端的 LoRA 配置处理，例如可以与基础模型或其他 LoRA 适配器的请求并行处理（前提是 `max_loras` 设置得足够大）。

请求示例：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## 动态加载 LoRA 适配器

除了在服务启动时配置 LoRA 适配器，vLLM 还支持通过专用 API 接口和插件（plugin）在运行时动态加载和卸载 LoRA 适配器。当你需要灵活切换模型时，这一功能尤其有用。

注意：在生产环境启用此功能存在一定风险，因为用户有可能参与适配器管理。

启用动态 LoRA 配置功能时，请确保环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 设置为 `True`。

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### 通过 API 接口加载/卸载

加载 LoRA 适配器：

要动态加载 LoRA 适配器，可以向 `/v1/load_lora_adapter` 发送 POST 请求，请求体中包含适配器的名称和路径。

示例请求：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

加载成功后，API 会返回 `200 OK`，`curl` 输出内容：`Success: LoRA adapter 'sql_adapter' added successfully`。如果出现错误（如找不到适配器或加载失败），会返回相应的错误提示。

卸载 LoRA 适配器：

要卸载已经加载的 LoRA 适配器，向 `/v1/unload_lora_adapter` 发送 POST 请求，包含需要卸载的适配器名称或 ID。

卸载成功后，API 返回 `200 OK`，`curl` 输出内容：`Success: LoRA adapter 'sql_adapter' removed successfully`。

示例请求：

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### 使用插件机制

你也可以通过 LoRAResolver 插件动态加载适配器。LoRAResolver 插件支持从本地文件系统或 S3 等本地和远程源加载适配器。当请求中出现新的模型名称时，LoRAResolver 会自动尝试加载相应适配器。

你可以配置多个 LoRAResolver 插件来从不同来源加载适配器。例如，可以为本地文件和 S3 分别设置解析器。vLLM 会优先加载第一个找到的适配器。

可以安装已有插件，也可以自行实现。vLLM 默认自带 [用于本地目录加载 LoRA 适配器的 resolver 插件](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)。启用方式：设置 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 为 True，`VLLM_PLUGINS` 包含 `lora_filesystem_resolver`，再将 `VLLM_LORA_RESOLVER_CACHE_DIR` 指定为本地目录。当请求中指定的适配器 `foobar` 在该本地目录下有同名子目录时，会自动加载该目录内容作为 LoRA 适配器。加载成功后，该适配器即可被正常调用。

如果你想自定义插件，可以参考以下步骤：

1. 实现 LoRAResolver 接口。

    ??? code "S3 LoRAResolver 简单实现示例"

        ```python
        import os
        import s3fs
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver

        class S3LoRAResolver(LoRAResolver):
            def __init__(self):
                self.s3 = s3fs.S3FileSystem()
                self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
                self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

            async def resolve_lora(self, base_model_name, lora_name):
                s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
                local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

                # 从 S3 下载 LoRA 到本地
                await self.s3._get(
                    s3_path, local_path, recursive=True, maxdepth=1
                )

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=local_path,
                    lora_int_id=abs(hash(lora_name)),
                )
                return lora_request
        ```

2. 注册 `LoRAResolver` 插件。

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    更多细节可参考 [vLLM 插件系统文档](../design/plugin_system.md)。

## `--lora-modules` 新格式说明

旧版本中，LoRA 模块通过 `name=path` 或 JSON 格式指定。例如：

```bash
--lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```

这种方式只包含 `name` 和 `path`，无法指定 `base_model_name`。现在可以使用 JSON 格式，额外指定 `base_model_name`，如下所示：

```bash
--lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

为了兼容旧格式，你仍然可以使用 `name=path`，只不过这种方式下 `base_model_name` 会保持未指定状态。

## 模型卡中的 LoRA 溯源关系

`--lora-modules` 的新格式主要是为了在模型卡片中支持父模型信息的显示。具体说明如下：

- LoRA 模型 `sql-lora` 的 `parent` 字段现在会指向其基础模型 `meta-llama/Llama-2-7b-hf`，准确反映两者的层次关系。
- `root` 字段会显示 LoRA 适配器的物理位置。

??? console "命令输出示例"

    ```bash
    $ curl http://localhost:8000/v1/models

    {
        "object": "list",
        "data": [
            {
            "id": "meta-llama/Llama-2-7b-hf",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/",
            "parent": null,
            "permission": [
                {
                .....
                }
            ]
            },
            {
            "id": "sql-lora",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "~/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
            "parent": meta-llama/Llama-2-7b-hf,
            "permission": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```

## 多模态模型的默认 LoRA

部分模型（如 [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) 和 [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) 等多模态模型），自带 LoRA 适配器，需要在特定模态信息存在时自动应用。如果按照前述方式操作，用户需要手动发送 `LoRARequest`（离线）或者根据请求内容区分基础模型与 LoRA 模型（服务端），使用起来较为繁琐。

为此，vLLM 支持注册默认的多模态 LoRA 适配器。你可以将每种模态映射到一个 LoRA 适配器，当输入中包含该模态时自动应用。注意，目前每条 prompt 只允许应用一个 LoRA，若输入同时包含多种已注册模态，则不会自动应用任何 LoRA。

??? code "离线推理示例"

    ```python
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    model_id = "ibm-granite/granite-speech-3.3-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_prompt(question: str, has_audio: bool):
        """构建要发送给 vLLM 的输入 prompt。"""
        if has_audio:
            question = f"<|audio|>{question}"
        chat = [
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False)


    llm = LLM(
        model=model_id,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
        # 只要请求中包含音频数据，就会自动传递 model_id 对应的 LoRARequest
        default_mm_loras = {"audio": model_id},
        enforce_eager=True,
    )

    question = "can you transcribe the speech into a written format?"
    prompt_with_audio = get_prompt(
        question=question,
        has_audio=True,
    )
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

    inputs = {
        "prompt": prompt_with_audio,
        "multi_modal_data": {
            "audio": audio,
        }
    }


    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=0.2,
            max_tokens=64,
        ),
    )
    ```

你也可以通过 `--default-mm-loras` 参数，传入一个 json 字典，将模态类型映射到 LoRA 模型 ID。例如，启动服务时：

```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --max-model-len 2048 \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

注意：默认的多模态 LoRA 目前仅支持 `.generate` 和聊天补全接口。

## 使用建议

### 配置 `max_lora_rank`

`--max-lora-rank` 参数用于设置 LoRA 适配器允许的最大秩（rank），这个设置会影响内存分配和性能：

- **建议设置为你所有 LoRA 适配器中最大的 rank**
- **避免设置过大** —— 远高于实际需求会浪费内存并可能影响性能

比如你的 LoRA 适配器 rank 分别是 [16, 32, 64]，那么应设置为 `--max-lora-rank 64`，而不是 256

```bash
# 推荐：与实际最大 rank 匹配
vllm serve model --enable-lora --max-lora-rank 64

# 不推荐：设置过大浪费内存
vllm serve model --enable-lora --max-lora-rank 256
```