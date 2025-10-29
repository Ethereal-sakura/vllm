# LiteLLM

[LiteLLM](https://github.com/BerriAI/litellm) 可以用 OpenAI 格式调用所有主流大模型（LLM）API，包括 Bedrock、Huggingface、VertexAI、TogetherAI、Azure、OpenAI、Groq 等

LiteLLM 主要功能包括：

- 自动将输入转换为不同厂商的 `completion`、`embedding` 和 `image_generation` 接口格式
- [统一的输出格式](https://docs.litellm.ai/docs/completion/output)，所有文本回复都可以通过 `['choices'][0]['message']['content']` 获取
- 支持多部署环境下（如 Azure/OpenAI）自动重试与故障切换 - 详见 [Router](https://docs.litellm.ai/docs/routing)
- 可为每个项目、API 密钥、模型设置预算和限流 [LiteLLM Proxy Server（LLM 网关）](https://docs.litellm.ai/docs/simple_proxy)

此外，LiteLLM 兼容所有 VLLM 上的模型。

## 前置条件

先搭建好 vLLM 和 litellm 的运行环境：

```bash
pip install vllm litellm
```

## 部署方法

### 聊天补全（Chat completion）

1. 首先用支持聊天补全的模型启动 vLLM 服务，例如：

    ```bash
    vllm serve qwen/Qwen1.5-0.5B-Chat
    ```

2. 然后用 litellm 调用 vLLM：

??? code

    ```python
    import litellm 

    messages = [{"content": "Hello, how are you?", "role": "user"}]

    # hosted_vllm 是必须的前缀关键字
    response = litellm.completion(
        model="hosted_vllm/qwen/Qwen1.5-0.5B-Chat", # 填写 vllm 的模型名称
        messages=messages,
        api_base="http://{your-vllm-server-host}:{your-vllm-server-port}/v1",
        temperature=0.2,
        max_tokens=80,
    )

    print(response)
    ```

### 向量嵌入（Embeddings）

1. 用支持 embedding 的模型启动 vLLM 服务，例如：

    ```bash
    vllm serve BAAI/bge-base-en-v1.5
    ```

2. 用 litellm 调用 vLLM：

```python
from litellm import embedding   
import os

os.environ["HOSTED_VLLM_API_BASE"] = "http://{your-vllm-server-host}:{your-vllm-server-port}/v1"

# hosted_vllm 是必须的前缀关键字
# 直接传 vllm 的模型名称
embedding = embedding(model="hosted_vllm/BAAI/bge-base-en-v1.5", input=["Hello world"])

print(embedding)
```

更多详细教程可以参考 [在 LiteLLM 中使用 vLLM](https://docs.litellm.ai/docs/providers/vllm)