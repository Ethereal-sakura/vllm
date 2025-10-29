# AnythingLLM

[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) 是一个全栈应用，可以将任何文档、资源或内容转化为上下文信息，让任何大语言模型（LLM）在聊天过程中引用这些内容。

它支持部署一个以 vLLM 为后端的大语言模型服务器，并提供兼容 OpenAI 的接口。

## 前提条件

先搭建 vLLM 运行环境：

```bash
pip install vllm
```

## 部署步骤

1. 使用支持聊天补全（chat-completion）的模型启动 vLLM 服务器，例如：

    ```bash
    vllm serve Qwen/Qwen1.5-32B-Chat-AWQ --max-model-len 4096
    ```

2. 下载并安装 [AnythingLLM Desktop](https://anythingllm.com/desktop)。

3. 配置 AI 提供方：

    - 在底部点击🔧扳手图标 -> **打开设置** -> **AI Providers** -> **LLM**。
    - 输入以下内容：
        - LLM Provider: Generic OpenAI
        - Base URL: `http://{vllm server host}:{vllm server port}/v1`
        - Chat Model Name: `Qwen/Qwen1.5-32B-Chat-AWQ`

    ![设置 AI 提供方](../../assets/deployment/anything-llm-provider.png)

4. 创建工作区：

    1. 在底部点击 ↺ 返回图标，回到工作区界面。
    2. 新建一个工作区（例如 `vllm`），即可开始对话。

    ![创建工作区](../../assets/deployment/anything-llm-chat-without-doc.png)

5. 添加文档。

    1. 点击 📎 附件图标。
    2. 上传你的文档。
    3. 选中并移动文档到你的工作区。
    4. 保存并嵌入文档。

    ![添加文档](../../assets/deployment/anything-llm-upload-doc.png)

6. 使用你的文档作为上下文进行聊天。

    ![使用文档上下文聊天](../../assets/deployment/anything-llm-chat-with-doc.png)