# AutoGen

[AutoGen](https://github.com/microsoft/autogen) 是一个用于创建多智能体（multi-agent）AI 应用框架，可以让 AI 自主行动，也可以协助人类完成任务。

## 前置条件

请先配置好 vLLM 和 [AutoGen](https://microsoft.github.io/autogen/0.2/docs/installation/) 运行环境：

```bash
pip install vllm

# 从 Extensions 安装 AgentChat 和 OpenAI 客户端
# AutoGen 需要 Python 3.10 或更高版本
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

## 部署

1. 使用支持聊天补全（chat completion）的模型启动 vLLM 服务，例如：

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.2
    ```

2. 使用 AutoGen 调用 vLLM 服务：

??? code

    ```python
    import asyncio
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelFamily


    async def main() -> None:
        # 创建模型客户端
        model_client = OpenAIChatCompletionClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            base_url="http://{your-vllm-host-ip}:{your-vllm-host-port}/v1",
            api_key="EMPTY",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.MISTRAL,
                "structured_output": True,
            },
        )

        messages = [UserMessage(content="写一个非常简短的关于龙的故事。", source="user")]

        # 创建流式对话
        stream = model_client.create_stream(messages=messages)

        # 遍历流式响应并打印结果
        print("流式返回内容：")
        async for response in stream:
            if isinstance(response, str):
                # 部分响应以字符串形式返回
                print(response, flush=True, end="")
            else:
                # 最后一次响应为包含完整消息的 CreateResult 对象
                print("\n\n------------\n")
                print("完整返回内容：", flush=True)
                print(response.content, flush=True)

        # 任务完成后关闭客户端
        await model_client.close()


    asyncio.run(main())
    ```

更多内容详见以下教程：

- [在 AutoGen 中使用 vLLM](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/local-vllm/)

- [兼容 OpenAI API 的使用示例](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html#autogen_ext.models.openai.OpenAIChatCompletionClient)