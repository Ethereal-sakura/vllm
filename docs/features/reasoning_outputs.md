# 推理输出

vLLM 支持推理（reasoning）模型，例如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，这些模型能够生成包含推理过程和最终结论的输出内容。

推理模型的输出中会多出一个 `reasoning_content` 字段，里面是模型推理得到最终结论的详细过程。这个字段在其他类型的模型输出里是没有的。

## 支持的模型

目前 vLLM 支持以下推理模型：

| 模型系列 | 解析器名称 | 结构化输出支持 | 工具调用 |
|--------------|-------------|------------------|-------------|
| [DeepSeek R1 系列](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `deepseek_r1` | `json`, `regex` | ❌ |
| [DeepSeek-V3.1](https://huggingface.co/collections/deepseek-ai/deepseek-v31-68a491bed32bd77e7fca048f) | `deepseek_v3` | `json`, `regex` | ❌ |
| [ERNIE-4.5-VL 系列](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT) | `ernie45` | `json`, `regex` | ❌ |
| [ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | `ernie45` | `json`, `regex` | ✅ |
| [GLM-4.5 系列](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) | `glm45` | `json`, `regex` | ✅ |
| [Hunyuan A13B 系列](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be) | `hunyuan_a13b` | `json`, `regex` | ✅ |
| [IBM Granite 3.2 语言模型](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a) | `granite` | ❌ | ❌ |
| [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) | `minimax_m2_append_think` | `json`, `regex` | ✅ |
| [Qwen3 系列](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `qwen3` | `json`, `regex` | ✅ |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `json`, `regex` | ✅ |

!!! note
    IBM Granite 3.2 和 DeepSeek-V3.1 的推理模式默认是关闭的，如果需要开启，还需在 `chat_template_kwargs` 中传入 `thinking=True`
    Qwen3 系列的推理功能默认开启，如果想关闭，则需传入 `enable_thinking=False` 到 `chat_template_kwargs`
    DeepSeek-V3.1 的工具调用仅支持非推理（non-thinking）模式

## 快速上手

如需使用推理模型，在调用聊天补全接口时需要通过 `--reasoning-parser` 参数指定推理内容解析器，用于从模型输出中提取推理内容。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

接下来，向模型发送请求，模型响应中会包含推理内容。

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和 API base，指向 vLLM 的 API 服务地址
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # 第一次对话
    messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]
    # 如果是 granite，需要加上：`extra_body={"chat_template_kwargs": {"thinking": True}}`
    # 如果是 Qwen3 系列，想关闭推理模式则加上：
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(model=model, messages=messages)

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print("reasoning_content:", reasoning_content)
    print("content:", content)
    ```

`reasoning_content` 字段包含了模型给出结论前的整个推理过程，而 `content` 字段就是最终的结论。

## 流式输出 chat completions

推理模型也支持流式聊天补全。在流式输出的 [chat completion response chunks](https://platform.openai.com/docs/api-reference/chat/streaming) 返回数据中，`reasoning_content` 会出现在 `delta` 字段里。

??? console "Json"

    ```json
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning_content": "is",
                },
                "logprobs": null,
                "finish_reason": null
            }
        ]
    }
    ```

OpenAI 的 Python 客户端库暂不支持流式输出中的 `reasoning_content` 属性，但客户端允许响应中包含自定义属性。你可以用 `hasattr` 判断响应中是否存在 `reasoning_content` 字段。例如：

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和 API base，指向 vLLM 的 API 服务地址
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]
    # 如果是 granite，需要加上：`extra_body={"chat_template_kwargs": {"thinking": True}}`
    # 如果是 Qwen3 系列，想关闭推理模式则加上：
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("client: 开始流式输出 chat completions ...")
    printed_reasoning_content = False
    printed_content = False

    for chunk in stream:
        # 安全地从 delta 中提取 reasoning_content 和 content，不存在时为 None
        reasoning_content = (
            getattr(chunk.choices[0].delta, "reasoning_content", None) or None
        )
        content = getattr(chunk.choices[0].delta, "content", None) or None

        if reasoning_content is not None:
            if not printed_reasoning_content:
                printed_reasoning_content = True
                print("reasoning_content:", end="", flush=True)
            print(reasoning_content, end="", flush=True)
        elif content is not None:
            if not printed_content:
                printed_content = True
                print("\ncontent:", end="", flush=True)
            # 输出最终内容
            print(content, end="", flush=True)
    ```

在访问 `reasoning_content` 前，请务必判断该字段是否存在。你可以参考 [example](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py) 里的代码示例。

## 工具调用（Tool Calling）

当工具调用和推理内容解析器同时启用时，推理内容同样会在响应中返回。另外，工具调用只会从 `content` 字段中解析函数调用，不会从 `reasoning_content` 里解析。

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定地点的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市和州，例如 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                }
            },
        }
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "旧金山的天气怎么样？"}],
        tools=tools,
        tool_choice="auto",
    )

    print(response)
    tool_call = response.choices[0].message.tool_calls[0].function

    print(f"reasoning_content: {response.choices[0].message.reasoning_content}")
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    ```

更多示例请参考 [examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py](../../examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py)

## 已知限制

- 推理内容目前仅支持在线服务的聊天补全接口（`/v1/chat/completions`）。

## 如何支持新的推理模型

你可以仿照 [vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py) 添加新的 `ReasoningParser`。

??? code

    ```python
    # 导入所需包

    from vllm.reasoning import ReasoningParser, ReasoningParserManager
    from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                                DeltaMessage)

    # 定义并注册推理解析器，register_module 里的名称可以在 --reasoning-parser 参数中使用
    @ReasoningParserManager.register_module(["example"])
    class ExampleParser(ReasoningParser):
        def __init__(self, tokenizer: AnyTokenizer):
            super().__init__(tokenizer)

        def extract_reasoning_content_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
        ) -> DeltaMessage | None:
            """
            实例方法，用于在流式响应时，从未完成的输出中提取推理内容。
            需要用到当前的 token 差异以及之前已解析和提取的内容（见构造函数）。
            """

        def extract_reasoning_content(
            self,
            model_output: str,
            request: ChatCompletionRequest | ResponsesRequest,
        ) -> tuple[str | None, str | None]:
            """
            从完整的模型输出字符串中提取推理内容。

            用于非流式响应，即在返回给客户端前，已经拿到完整的模型输出。

            参数说明：
            model_output: str
                需要提取推理内容的模型输出字符串

            request: ChatCompletionRequest
                生成该输出时用的请求对象

            返回值:
            tuple[Optional[str], Optional[str]]
                返回推理内容和最终内容的二元组
            """
    ```

如果需要支持结构化输出，还需新增一个 `Reasoner`，可参考 [vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py) 的实现。

??? code

    ```python
    @dataclass
    class DeepSeekReasoner(Reasoner):
        """
        DeepSeek R 系列模型的推理器。
        """
        start_token_id: int
        end_token_id: int

        start_token: str = "<think>"
        end_token: str = "</think>"

        @classmethod
        def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
            return cls(
                start_token_id=tokenizer.encode("<think>", add_special_tokens=False)[0],
                end_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
            )

        def is_reasoning_end(self, input_ids: list[int]) -> bool:
            return self.end_token_id in input_ids
        ...
    ```

像 [xgrammar](https://github.com/mlc-ai/xgrammar) 这样的结构化输出引擎会通过 `end_token_id` 判断模型输出中是否包含推理内容，并在需要时跳过结构化输出。

最后，通过 `--reasoning-parser` 参数即可为模型启用推理功能：

```bash
vllm serve <model_tag> --reasoning-parser example
```