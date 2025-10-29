# Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) 是一个可扩展、功能丰富且易于使用的本地自部署 AI 平台，支持完全离线运行。它兼容多种大语言模型（LLM）运行框架，如 Ollama 以及兼容 OpenAI 的 API，并内置 RAG（检索增强生成）能力，是一套强大的 AI 部署解决方案。

如果你希望通过 vLLM 快速体验 Open WebUI，可以按照以下步骤操作：

1. 安装 [Docker](https://docs.docker.com/engine/install/)。

2. 使用受支持的 chat completion 模型启动 vLLM 服务器：

    ```console
    vllm serve Qwen/Qwen3-0.6B-Chat
    ```

    !!! note
        启动 vLLM 服务器时，请务必通过 `--host` 和 `--port` 参数指定主机和端口。例如：

        ```console
        vllm serve <model> --host 0.0.0.0 --port 8000
        ```

3. 启动 Open WebUI 的 Docker 容器：

    ```console
    docker run -d \
        --name open-webui \
        -p 3000:8080 \
        -v open-webui:/app/backend/data \
        -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
        --restart always \
        ghcr.io/open-webui/open-webui:main
    ```

4. 在浏览器中打开：<http://open-webui-host:3000/>

    页面顶部应可以看到模型 `Qwen/Qwen3-0.6B-Chat`。

    ![Web portal of model Qwen/Qwen3-0.6B-Chat](../../assets/deployment/open_webui.png)
