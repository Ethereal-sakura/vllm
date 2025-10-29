# Chatbox

[Chatbox](https://github.com/chatboxai/chatbox) 是一款适用于 Windows、Mac 和 Linux 的桌面端大语言模型（LLM）客户端。

它支持将 vLLM 作为后端，快速搭建大语言模型服务，并提供与 OpenAI 兼容的接口。

## 前置条件

请先搭建 vLLM 环境：

```bash
pip install vllm
```

## 部署流程

1. 使用支持对话补全的模型启动 vLLM 服务，例如：

    ```bash
    vllm serve qwen/Qwen1.5-0.5B-Chat
    ```

1. 下载并安装 [Chatbox 桌面客户端](https://chatboxai.app/en#download)。

1. 在设置界面左下角，添加自定义服务商
    - API 模式：`OpenAI API Compatible`
    - 名称：vllm
    - API Host：`http://{vllm server host}:{vllm server port}/v1`
    - API Path：`/chat/completions`
    - 模型：`qwen/Qwen1.5-0.5B-Chat`

    ![](../../assets/deployment/chatbox-settings.png)

1. 进入 `Just chat` 页面，开始与你的模型对话：

    ![](../../assets/deployment/chatbox-chat.png)
