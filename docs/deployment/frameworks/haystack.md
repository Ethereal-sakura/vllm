# Haystack

[Haystack](https://github.com/deepset-ai/haystack) 是一个端到端的大语言模型（LLM）框架，可以帮助你构建由 LLM、Transformer 模型、向量检索等技术驱动的应用。不论你想实现检索增强生成（RAG）、文档搜索、问答还是答案生成，Haystack 都能将最先进的嵌入模型和 LLM 通过流水线（pipeline）组合起来，打造完整的自然语言处理（NLP）应用，满足你的实际需求。

它支持部署以 vLLM 作为后端的大语言模型服务器，并提供与 OpenAI 兼容的接口。

## 前置条件

请先搭建好 vLLM 和 Haystack 的运行环境：

```bash
pip install vllm haystack-ai
```

## 部署

1. 启动支持对话生成的 vLLM 服务器，例如：

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.1
    ```

2. 在 Haystack 中使用 `OpenAIGenerator` 和 `OpenAIChatGenerator` 组件来访问 vLLM 服务器。

??? code

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    generator = OpenAIChatGenerator(
        # 为了兼容 OpenAI API，这里需要一个占位的 api_key
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model="mistralai/Mistral-7B-Instruct-v0.1",
        api_base_url="http://{your-vLLM-host-ip}:{your-vLLM-host-port}/v1",
        generation_kwargs={"max_tokens": 512},
    )

    response = generator.run(
      messages=[ChatMessage.from_user("Hi. Can you help me plan my next trip to Italy?")]
    )

    print("-"*30)
    print(response)
    print("-"*30)
    ```

```console
------------------------------
{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=' Of course! Where in Italy would you like to go and what type of trip are you looking to plan?')], _name=None, _meta={'model': 'mistralai/Mistral-7B-Instruct-v0.1', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 23, 'prompt_tokens': 21, 'total_tokens': 44, 'completion_tokens_details': None, 'prompt_tokens_details': None}})]}
------------------------------
```

更多细节可以参考教程 [在 Haystack 中使用 vLLM](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/vllm.md)