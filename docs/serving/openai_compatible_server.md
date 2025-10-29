# 兼容 OpenAI 的服务端

vLLM 提供了一个 HTTP 服务端，实现了 OpenAI 的 [Completions API](https://platform.openai.com/docs/api-reference/completions)、[Chat API](https://platform.openai.com/docs/api-reference/chat) 等多种 API！通过这个功能，你可以部署模型，并通过 HTTP 客户端与其交互。

你可以在终端中先[安装](../getting_started/installation/README.md) vLLM，然后使用 [`vllm serve`](../configuration/serve_args.md) 命令启动服务端。（你也可以使用我们的 [Docker](../deployment/docker.md) 镜像。）

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

调用服务端时，可以在你喜欢的文本编辑器中编写一个脚本，使用 HTTP 客户端，并写入你想要发送给模型的消息。运行脚本即可。下面是一个使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 的示例脚本。

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion.choices[0].message)
    ```

!!! tip
    vLLM 支持一些 OpenAI 没有的参数，比如 `top_k`。
    你可以通过 OpenAI 客户端的 `extra_body` 字段，把这些参数传递给 vLLM，例如 `extra_body={"top_k": 50}`。

!!! important
    默认情况下，如果 Hugging Face 模型仓库中存在 `generation_config.json`，服务端会自动加载它。这意味着部分采样参数的默认值可能会被模型作者推荐的值覆盖。

    如果你希望禁用这个行为，启动服务端时请加上 `--generation-config vllm` 参数。

## 支持的 API

目前我们支持以下 OpenAI API：

- [Completions API](#completions-api)（`/v1/completions`）
    - 仅适用于[文本生成模型](../models/generative_models.md)。
    - *注意：暂不支持 `suffix` 参数。*
- [Chat Completions API](#chat-api)（`/v1/chat/completions`）
    - 仅适用于带有[聊天模板](../serving/openai_compatible_server.md#chat-template)的[文本生成模型](../models/generative_models.md)。
    - *注意：`parallel_tool_calls` 和 `user` 参数会被忽略。*
- [Embeddings API](#embeddings-api)（`/v1/embeddings`）
    - 仅适用于[Embedding 模型](../models/pooling_models.md)。
- [Transcriptions API](#transcriptions-api)（`/v1/audio/transcriptions`）
    - 仅适用于[自动语音识别（ASR）模型](../models/supported_models.md#transcription)。
- [Translation API](#translations-api)（`/v1/audio/translations`）
    - 仅适用于[自动语音识别（ASR）模型](../models/supported_models.md#transcription)。

此外，我们还提供以下自定义 API：

- [Tokenizer API](#tokenizer-api)（`/tokenize`, `/detokenize`）
    - 适用于所有带有分词器的模型。
- [Pooling API](#pooling-api)（`/pooling`）
    - 适用于所有[池化模型](../models/pooling_models.md)。
- [Classification API](#classification-api)（`/classify`）
    - 仅适用于[分类模型](../models/pooling_models.md)。
- [Score API](#score-api)（`/score`）
    - 适用于[Embedding 模型和交叉编码器模型](../models/pooling_models.md)。
- [Re-rank API](#re-rank-api)（`/rerank`, `/v1/rerank`, `/v2/rerank`）
    - 实现了 [Jina AI v1 re-rank API](https://jina.ai/reranker/)
    - 同时兼容 [Cohere v1 & v2 re-rank API](https://docs.cohere.com/v2/reference/rerank)
    - Jina 和 Cohere 的 API 十分相似，Jina 的接口会在响应中带有额外信息。
    - 仅适用于[交叉编码器模型](../models/pooling_models.md)。

## 聊天模板（Chat Template）

为了让语言模型支持聊天协议，vLLM 要求模型的分词器配置中包含聊天模板（chat template）。聊天模板是一个 Jinja2 模板，规定了角色、消息及其他聊天相关 token 如何被编码到输入中。

`NousResearch/Meta-Llama-3-8B-Instruct` 的聊天模板示例可以在 [这里](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models) 查看。

有些模型虽然进行了指令/聊天微调，但并没有内置聊天模板。对于这些模型，你可以通过 `--chat-template` 参数手动指定聊天模板文件路径，或者直接传递模板字符串。如果没有聊天模板，服务端将无法处理聊天请求，所有相关请求都会报错。

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM 社区为主流模型准备了一些聊天模板，你可以在 [examples](../../examples) 目录下找到。

随着多模态聊天 API 的加入，OpenAI 规范现在支持以新格式传递聊天消息，即同时指定 `type` 和 `text` 字段。如下示例所示：

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"},
            ],
        },
    ],
)
```

大多数 LLM 的聊天模板要求 `content` 字段为字符串，但也有部分新模型（如 `meta-llama/Llama-Guard-3-1B`）要求内容按 OpenAI 的 schema 格式化。vLLM 会自动检测并适配（日志中会显示类似 *"Detected the chat template content format to be..."*），自动将请求内容转换为模型需要的格式。格式可能如下：

- `"string"`：普通字符串。
    - 例如：`"Hello world"`
- `"openai"`：字典列表，类似 OpenAI schema。
    - 例如：`[{"type": "text", "text": "Hello world!"}]`

如果自动检测不是你想要的，可以通过 `--chat-template-content-format` CLI 参数手动指定格式。

## 额外参数

vLLM 支持一组 OpenAI API 没有的参数。
你可以在 OpenAI 客户端中通过 `extra_body` 参数传递这些参数，
或者直接在 HTTP 请求的 JSON 载荷中加入这些参数。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## 额外 HTTP 头

目前仅支持 `X-Request-Id` HTTP 请求头。可以通过 `--enable-request-id-headers` 参数启用。

??? code

    ```python
    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
        ],
        extra_headers={
            "x-request-id": "sentiment-classification-00001",
        },
    )
    print(completion._request_id)

    completion = client.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        prompt="A robot may not injure a human being",
        extra_headers={
            "x-request-id": "completion-test",
        },
    )
    print(completion._request_id)
    ```

## API 参考

### Completions API

我们的 Completions API 与 [OpenAI 的 Completions API](https://platform.openai.com/docs/api-reference/completions) 兼容；
你可以直接使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 调用。

代码示例：[examples/online_serving/openai_completion_client.py](../../examples/online_serving/openai_completion_client.py)

#### 额外参数

下列[采样参数](../api/README.md#inference-parameters)可用。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:completion-sampling-params"
    ```

此外，还支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:completion-extra-params"
    ```

### Chat API

我们的 Chat API 与 [OpenAI 的 Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 兼容；
你可以直接使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 调用。

我们同时支持 [Vision](https://platform.openai.com/docs/guides/vision) 和
[Audio](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 相关参数；
更多信息请参考[多模态输入](../features/multimodal_inputs.md)指南。

- *注意：暂不支持 `image_url.detail` 参数。*

代码示例：[examples/online_serving/openai_chat_completion_client.py](../../examples/online_serving/openai_chat_completion_client.py)

#### 额外参数

下列[采样参数](../api/README.md#inference-parameters)可用。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:chat-completion-sampling-params"
    ```

此外，还支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:chat-completion-extra-params"
    ```

### Embeddings API

我们的 Embeddings API 与 [OpenAI 的 Embeddings API](https://platform.openai.com/docs/api-reference/embeddings) 兼容；
你可以直接使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 调用。

代码示例：[examples/online_serving/pooling/openai_embedding_client.py](../../examples/online_serving/pooling/openai_embedding_client.py)

如果模型有[聊天模板](../serving/openai_compatible_server.md#chat-template)，你可以用 `messages` 列表（格式同 [Chat API](#chat-api)）替代 `inputs`，会把这组消息当作单个 prompt 发送给模型。以下是一个带有 OpenAI 类型注解的便捷调用方式：

??? code

    ```python
    from openai import OpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.create_embedding_response import CreateEmbeddingResponse

    def create_chat_embeddings(
        client: OpenAI,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        return client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={"messages": messages, "model": model, "encoding_format": encoding_format},
        )
    ```

#### 多模态输入

你可以自定义聊天模板，并通过请求中的 `messages` 字段，向 embedding 模型传递多模态输入。具体用法见下方示例。

=== "VLM2Vec"

    启动模型服务：

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
      --trust-remote-code \
      --max-model-len 4096 \
      --chat-template examples/template_vlm2vec_phi3v.jinja
    ```

    !!! important
        由于 VLM2Vec 与 Phi-3.5-Vision 架构相同，我们需要显式传递 `--runner pooling`，以便以 embedding 模式运行，而不是文本生成模式。

        此模型的自定义聊天模板与原模板完全不同，可在此处查看：[examples/template_vlm2vec_phi3v.jinja](../../examples/template_vlm2vec_phi3v.jinja)

    由于请求格式不由 OpenAI 客户端定义，我们可以用更底层的 `requests` 库向服务端发起请求：

    ??? code

        ```python
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        response = create_chat_embeddings(
            client,
            model="TIGER-Lab/VLM2Vec-Full",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }
            ],
            encoding_format="float",
        )

        print("Image embedding output:", response.data[0].embedding)
        ```

=== "DSE-Qwen2-MRL"

    启动模型服务：

    ```bash
    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --runner pooling \
      --trust-remote-code \
      --max-model-len 8192 \
      --chat-template examples/template_dse_qwen2_vl.jinja
    ```

    !!! important
        同样需要显式传递 `--runner pooling`。

        此外，`MrLight/dse-qwen2-2b-mrl-v1` 需要在 embedding 时加上 EOS token，这由自定义聊天模板处理：[examples/template_dse_qwen2_vl.jinja](../../examples/template_dse_qwen2_vl.jinja)

    !!! important
        `MrLight/dse-qwen2-2b-mrl-v1` 在文本查询 embedding 时需要传入最小尺寸的占位图像，完整代码示例请见下文。

完整代码示例：[examples/online_serving/pooling/openai_chat_embedding_client_for_multimodal.py](../../examples/online_serving/pooling/openai_chat_embedding_client_for_multimodal.py)

#### 额外参数

支持以下 [pooling 参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embedding-pooling-params"
```

默认还支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:embedding-extra-params"
    ```

若输入为聊天格式（即传入了 `messages`），则支持下列额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/protocol.py:chat-embedding-extra-params"
    ```

### Transcriptions API

我们的 Transcriptions API 与 [OpenAI 的 Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription) 兼容；
你可以直接使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 调用。

!!! note
    若需使用 Transcriptions API，请使用 `pip install vllm[audio]` 安装额外的音频依赖。

代码示例：[examples/online_serving/openai_transcription_client.py](../../examples/online_serving/openai_transcription_client.py)

#### API 限制

你可以通过环境变量 `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` 设置 vLLM 接受的最大音频文件大小（单位 MB），默认值为 25 MB。

#### 上传音频文件

Transcriptions API 支持上传多种格式的音频文件，包括 FLAC、MP3、MP4、MPEG、MPGA、M4A、OGG、WAV、WEBM 等。

**使用 OpenAI Python 客户端：**

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # 从磁盘上传音频文件
    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

    print(transcription.text)
    ```

**使用 curl 与 multipart/form-data：**

??? code

    ```bash
    curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
      -H "Authorization: Bearer token-abc123" \
      -F "file=@audio.mp3" \
