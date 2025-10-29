# 生成式模型

vLLM 对生成式模型（Generative Models）提供了原生支持，涵盖了大多数大语言模型（LLM）。

在 vLLM 中，生成式模型需要实现 [VllmModelForTextGeneration][vllm.model_executor.models.VllmModelForTextGeneration] 接口。  
这些模型会根据输入的最终隐藏状态输出生成 token 的对数概率（log probabilities），  
随后通过 [Sampler][vllm.v1.sample.sampler.Sampler] 采样，得到最终的文本结果。

## 配置

### 模型运行模式（`--runner`）

通过 `--runner generate` 选项，可以让模型以生成模式运行。

!!! tip
    绝大多数情况下无需手动设置该选项，vLLM 能自动根据模型判断应选择的运行模式，默认使用 `--runner auto`。

## 离线推理

[LLM][vllm.LLM] 类为离线推理场景提供了多种方法。  
关于模型初始化时可用的选项，请参考 [配置文档](../api/README.md#configuration)。

### `LLM.generate`

所有 vLLM 支持的生成式模型都可以使用 [generate][vllm.LLM.generate] 方法。  
其用法类似于 [HF Transformers 的 generate 方法](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)，  
不同之处在于，vLLM 会自动完成分词（tokenization）和反分词（detokenization）操作。

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("Hello, my name is")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

你可以通过传入 [SamplingParams][vllm.SamplingParams] 参数，自定义文本生成方式。  
例如，将 `temperature=0` 设置为贪婪采样（greedy sampling）：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! important
    默认情况下，vLLM 会自动加载 Huggingface 模型仓库中的 `generation_config.json`，采用模型开发者推荐的采样参数。通常，如果不显式指定 [SamplingParams][vllm.SamplingParams]，你会直接获得最佳的生成效果。

    若你更倾向于使用 vLLM 的默认采样参数，可以在创建 [LLM][vllm.LLM] 实例时传入 `generation_config="vllm"`。
    代码示例见：[examples/offline_inference/basic/basic.py](../../examples/offline_inference/basic/basic.py)

### `LLM.beam_search`

[beam_search][vllm.LLM.beam_search] 方法基于 [generate][vllm.LLM.generate] 实现了 [束搜索（beam search）](https://huggingface.co/docs/transformers/en/generation_strategies#beam-search)。  
例如，使用 5 个 beam 并最多生成 50 个 token：

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")
params = BeamSearchParams(beam_width=5, max_tokens=50)
outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)

for output in outputs:
    generated_text = output.sequences[0].text
    print(f"Generated text: {generated_text!r}")
```

### `LLM.chat`

[chat][vllm.LLM.chat] 方法是在 [generate][vllm.LLM.generate] 基础上实现的对话功能。  
它的输入格式与 [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 类似，  
并会自动应用模型的 [chat template（对话模板）](https://huggingface.co/docs/transformers/en/chat_templating) 来格式化 prompt。

!!! important
    通常，只有经过指令微调（instruction-tuned）的模型才包含对话模板。  
    基础模型（base model）由于没有经过对话训练，直接用于 chat 效果可能不佳。

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

相关代码示例可参考：[examples/offline_inference/basic/chat.py](../../examples/offline_inference/basic/chat.py)

如果模型没有内置对话模板，或者你希望使用自定义模板，  
可以通过参数显式指定 chat template：

```python
from vllm.entrypoints.chat_utils import load_chat_template

# 已有的 chat template 可以在 `examples/` 目录下找到
custom_template = load_chat_template(chat_template="<path_to_template>")
print("Loaded chat template:", custom_template)

outputs = llm.chat(conversation, chat_template=custom_template)
```

## 在线服务

我们的 [OpenAI 兼容服务端](../serving/openai_compatible_server.md) 提供了与离线 API 对应的服务接口：

- [Completions API](../serving/openai_compatible_server.md#completions-api) 与 `LLM.generate` 类似，但只接受文本输入
- [Chat API](../serving/openai_compatible_server.md#chat-api) 与 `LLM.chat` 类似，支持文本和 [多模态输入](../features/multimodal_inputs.md)（适用于有 chat template 的模型）
