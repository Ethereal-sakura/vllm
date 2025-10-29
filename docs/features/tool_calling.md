# 工具调用（Tool Calling）

vLLM 目前支持命名函数调用，并且在 chat completion API 的 `tool_choice` 字段中，还支持 `auto`、`required`（自 `vllm>=0.8.3` 起）、以及 `none` 选项。

## 快速开始

启动服务器并开启工具调用功能。本示例使用 Meta 的 Llama 3.1 8B 模型，因此需要使用 vLLM 示例目录中的 `llama3_json` 工具调用聊天模板：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

接下来，发送一个请求，触发模型调用可用工具：

??? code

    ```python
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    def get_weather(location: str, unit: str):
        return f"Getting the weather for {location} in {unit}..."
    tool_functions = {"get_weather": get_weather}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定位置的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市和州，例如 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location", "unit"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
    ```

示例输出：

```text
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

该示例演示了以下流程：

* 启用工具调用功能启动服务器
* 定义实际的函数来处理工具调用
* 使用 `tool_choice="auto"` 方式发起请求
* 处理结构化响应并执行对应函数

你也可以通过设置 `tool_choice={"type": "function", "function": {"name": "get_weather"}}` 来指定某个具体函数进行调用。需要注意的是，这种方式会使用结构化输出后端，因此首次使用时会有几秒甚至更长的延迟，因为 FSM 会首次编译并缓存，之后请求会更快。

请记住，调用方需要负责：

1. 在请求中定义合适的工具
2. 在聊天消息中提供相关上下文
3. 在应用逻辑里处理工具调用

如需更高级的用法，包括并行工具调用和不同模型专用解析器，请参考下方相关章节。

## 命名函数调用

vLLM 默认支持在 chat completion API 中进行命名函数调用。绝大多数 vLLM 支持的结构化输出后端都可以正常工作。你可以获得格式正确、可解析的函数调用——但并不保证调用质量很高。

vLLM 会用结构化输出确保响应内容符合在 `tools` 参数中通过 JSON schema 定义的工具参数对象。为获得最佳效果，建议在 prompt 中指定期望的输出格式或 schema，这样模型生成内容时会更贴合结构化输出后端要求的 schema。

要使用指定的函数，只需在 chat completion 请求的 `tools` 参数中定义函数，并在 `tool_choice` 参数里指定要调用的工具 `name`。

## 必须调用函数

vLLM 支持在 chat completion API 中使用 `tool_choice='required'` 选项。和命名函数调用类似，内部也是用结构化输出，因此默认启用，所有支持的模型都能用。未来也会支持其他解码后端，具体进展可参考[路线图](../usage/v1_guide.md#features)。

当设置 `tool_choice='required'` 时，模型会根据 `tools` 参数中的工具列表，保证生成一个或多个工具调用，具体数量取决于用户的请求。输出格式会严格遵循 `tools` 参数中定义的 schema。

## 禁用工具调用

vLLM 支持在 chat completion API 中将 `tool_choice='none'`，此时模型不会生成任何工具调用，只会返回普通文本内容，即便请求中定义了工具。

!!! note
    如果请求中有定义工具，vLLM 默认会将工具定义加入 prompt，无论 `tool_choice` 是否设置为 `none`。如果希望在 `tool_choice='none'` 时排除工具定义，可以使用 `--exclude-tools-when-tool-choice-none` 选项。

## 自动工具调用

启用自动工具调用功能时需要设置以下参数：

* `--enable-auto-tool-choice` —— **必需**，自动工具选择，允许模型在它认为需要时自动生成工具调用。
* `--tool-call-parser` —— 选择工具解析器（见下方列表）。未来会不断增加更多工具解析器，也可以用 `--tool-parser-plugin` 注册自定义解析器。
* `--tool-parser-plugin` —— **可选**，用于注册用户自定义工具解析器到 vLLM，注册后可以在 `--tool-call-parser` 指定名称。
* `--chat-template` —— **可选**，用于自动工具选择时指定聊天模板，模板会处理 `tool` 角色消息和 `assistant` 角色消息里的工具调用。Hermes、Mistral 和 Llama 等模型在它们的 `tokenizer_config.json` 文件中已包含兼容工具调用的聊天模板，当然也可以指定自定义模板。如果你的模型有专用的工具调用聊天模板，可以将此参数设为 `tool_use`，此时会按 `transformers` 规范使用。更多信息可参考 [HuggingFace 文档](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)；示例见 [tokenizer_config.json](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)。

如果你喜欢的工具调用模型还不支持，欢迎贡献解析器和聊天模板！

### Hermes 模型（`hermes`）

所有 Nous Research Hermes 系列中 Hermes 2 Pro 及之后的模型都支持。

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_注意 Hermes 2 **Theta** 系列由于合并步骤影响，工具调用质量和能力有所下降。_

参数：`--tool-call-parser hermes`

### Mistral 模型（`mistral`）

支持的模型：

* `mistralai/Mistral-7B-Instruct-v0.3`（已确认）
* 其他支持函数调用的 Mistral 系列模型也兼容。

已知问题：

1. Mistral 7B 在生成并行工具调用时有困难。
2. **仅限 Transformers 分词后端**：Mistral 的 `tokenizer_config.json` 聊天模板要求工具调用 ID 必须是 9 位数字，而 vLLM 生成的更长。如果不满足会报错，建议使用以下模板：

    * [examples/tool_chat_template_mistral.jinja](../../examples/tool_chat_template_mistral.jinja) —— 官方聊天模板，已调整为支持 vLLM 的工具调用 ID（会截取最后 9 位）。
    * [examples/tool_chat_template_mistral_parallel.jinja](../../examples/tool_chat_template_mistral_parallel.jinja) —— 进一步增强，并在有工具时添加工具用法提示，提升并行工具调用的稳定性。

推荐参数：

1. 使用官方 Mistral 分词后端 [mistral-common](https://github.com/mistralai/mistral-common)：

    `--tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral`

2. 使用默认 Transformers 分词后端：
    `--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 模型（`llama3_json`）

支持的模型：

所有 Llama 3.1、3.2 和 4 系列均支持。

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

支持的工具调用格式为 [基于 JSON 的工具调用](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)。对于 Llama-3.2 新增的 [Pythonic 工具调用](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling)，请参考下方 `pythonic` 工具解析器。Llama 4 推荐使用 `llama4_pythonic` 工具解析器。

其他如内置 python 工具调用或自定义工具调用暂不支持。

已知问题：

1. Llama 3 不支持并行工具调用，但 Llama 4 支持。
2. 模型有时会生成参数格式错误，比如把数组序列化为字符串而不是数组。

vLLM 针对 Llama 3.1 和 3.2 提供了两套 JSON 聊天模板：

* [examples/tool_chat_template_llama3.1_json.jinja](../../examples/tool_chat_template_llama3.1_json.jinja) ——这是 Llama 3.1 官方模板的增强版，更适配 vLLM。
* [examples/tool_chat_template_llama3.2_json.jinja](../../examples/tool_chat_template_llama3.2_json.jinja) ——在 3.1 基础上增加了图像支持。

推荐参数：`--tool-call-parser llama3_json --chat-template {见上方}`

Llama 4 还提供了 Pythonic 与 JSON 聊天模板，推荐使用 Pythonic 工具调用：

* [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) ——基于 [官方聊天模板](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)。

Llama 4 推荐使用 `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja`。

### IBM Granite

支持的模型：

* `ibm-granite/granite-4.0-h-small` 及其他 Granite 4.0 系列

    推荐参数：`--tool-call-parser hermes`

* `ibm-granite/granite-3.0-8b-instruct`

    推荐参数：`--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

    [examples/tool_chat_template_granite.jinja](../../examples/tool_chat_template_granite.jinja)：此模板基于 Hugging Face 原版修改，支持并行函数调用。

* `ibm-granite/granite-3.1-8b-instruct`

    推荐参数：`--tool-call-parser granite`

    可直接使用 Huggingface 的聊天模板，支持并行函数调用。

* `ibm-granite/granite-20b-functioncalling`

    推荐参数：`--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

    [examples/tool_chat_template_granite_20b_fc.jinja](../../examples/tool_chat_template_granite_20b_fc.jinja)：此模板基于 Hugging Face 原版修改，融合 Hermes 模板的函数描述元素，并遵循论文中“Response Generation”模式的系统提示。支持并行函数调用。

### InternLM 模型（`internlm`）

支持的模型：

* `internlm/internlm2_5-7b-chat`（已确认）
* 其他支持函数调用的 internlm2.5 模型也兼容

已知问题：

* 该实现也支持 InternLM2，但在 `internlm/internlm2-chat-7b` 测试时工具调用结果不够稳定。

推荐参数：`--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba 模型（`jamba`）

AI21 的 Jamba-1.5 系列模型已支持。

* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`

参数：`--tool-call-parser jamba`

### xLAM 模型（`xlam`）

xLAM 工具解析器适用于多种 JSON 格式工具调用模型，能识别多种输出风格：

1. 直接 JSON 数组：输出字符串是以 `[` 开头、`]` 结尾的 JSON 数组
2. 思考标签：使用 `<think>...</think>` 标签包裹 JSON 数组
3. 代码块：在代码块 (```json ...```) 中的 JSON
4. 工具调用标签：用 `[TOOL_CALLS]` 或 `<tool_call>...</tool_call>` 标签

支持并行函数调用，且能有效区分文本内容与工具调用。

已支持模型：

* Salesforce Llama-xLAM 系列：`Salesforce/Llama-xLAM-2-8B-fc-r`、`Salesforce/Llama-xLAM-2-70B-fc-r`
* Qwen-xLAM 系列：`Salesforce/xLAM-1B-fc-r`、`Salesforce/xLAM-3B-fc-r`、`Salesforce/Qwen-xLAM-32B-fc-r`

参数：

* Llama 系列 xLAM：`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_llama.jinja`
* Qwen 系列 xLAM：`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_qwen.jinja`

### Qwen 模型

对于 Qwen2.5，tokenizer_config.json 内置了 Hermes 风格工具调用支持，因此可以用 `hermes` 解析器开启工具调用。更多详情请参考官方 [Qwen 文档](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)

* `Qwen/Qwen2.5-*`
* `Qwen/QwQ-32B`

参数：`--tool-call-parser hermes`

### MiniMax 模型（`minimax_m1`）

支持的模型：

* `MiniMaxAi/MiniMax-M1-40k`（须搭配 [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja) 使用）
* `MiniMaxAi/MiniMax-M1-80k`（须搭配 [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja) 使用）

参数：`--tool-call-parser minimax --chat-template examples/tool_chat_template_minimax_m1.jinja`

### DeepSeek-V3 模型（`deepseek_v3`）

支持的模型：

* `deepseek-ai/DeepSeek-V3-0324`（须搭配 [examples/tool_chat_template_deepseekv3.jinja](../../examples/tool_chat_template_deepseekv3.jinja) 使用）
* `deepseek-ai/DeepSeek-R1-0528`（须搭配 [examples/tool_chat_template_deepseekr1.jinja](../../examples/tool_chat_template_deepseekr1.jinja) 使用）

参数：`--tool-call-parser deepseek_v3 --chat-template {见上方}`

### DeepSeek-V3.1 模型（`deepseek_v31`）

支持的模型：

* `deepseek-ai/DeepSeek-V3.1`（须搭配 [examples/tool_chat_template_deepseekv31.jinja](../../examples/tool_chat_template_deepseekv31.jinja) 使用）

参数：`--tool-call-parser deepseek_v31 --chat-template {见上方}`

### Kimi-K2 模型（`kimi_k2`）

支持的模型：

* `moonshotai/Kimi-K2-Instruct`

参数：`--tool-call-parser kimi_k2`

### Hunyuan 模型（`hunyuan_a13b`）

支持的模型：

* `tencent/Hunyuan-A13B-Instruct`（聊天模板已包含在 Hugging Face 模型文件中）

参数：

* 非推理模式：`--tool-call-parser hunyuan_a13b`
* 推理模式：`--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`

### LongCat-Flash-Chat 模型（`longcat`）

支持的模型：

* `meituan-longcat/LongCat-Flash-Chat`
* `meituan-longcat/LongCat-Flash-Chat-FP8`

参数：`--tool-call-parser longcat`

### GLM-4.5 模型（`glm45`）

支持的模型：

* `zai-org/GLM-4.5`
* `zai-org/GLM-4.5-Air`
* `zai-org/GLM-4.6`
* `zai-org/GLM-4.6-Air`

参数：`--tool-call-parser glm45`

### Qwen3-Coder 模型（`qwen3_xml`）

支持的模型：

* `Qwen/Qwen3-480B-A35B-Instruct`
* `