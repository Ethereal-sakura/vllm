# Dify

[Dify](https://github.com/langgenius/dify) 是一个开源的大语言模型（LLM）应用开发平台。它拥有直观的操作界面，集成了智能体（agentic AI）工作流、RAG 管道、智能体能力、模型管理、可观测性等多项功能，帮助你快速将原型转化为生产应用。

Dify 支持将 vLLM 作为模型服务，能够高效部署大语言模型。

本指南将带你完成基于 vLLM 部署 Dify 的全部流程。

## 前置准备

先搭建 vLLM 环境：

```bash
pip install vllm
```

并安装 [Docker](https://docs.docker.com/engine/install/) 及 [Docker Compose](https://docs.docker.com/compose/install/)。

## 部署步骤

1. 使用支持的聊天补全模型启动 vLLM 服务，例如：

    ```bash
    vllm serve Qwen/Qwen1.5-7B-Chat
    ```

2. 使用 docker compose 启动 Dify 服务（[详细步骤](https://github.com/langgenius/dify?tab=readme-ov-file#quick-start)）：

    ```bash
    git clone https://github.com/langgenius/dify.git
    cd dify
    cd docker
    cp .env.example .env
    docker compose up -d
    ```

3. 在浏览器中访问 `http://localhost/install`，配置基础的登录信息并完成登录。

4. 点击右上角个人头像进入用户菜单，选择“设置”，点击 `Model Provider`，找到 `vLLM` 服务商进行安装。

5. 按如下方式填写模型服务信息：

    - **Model Type**：`LLM`
    - **Model Name**：`Qwen/Qwen1.5-7B-Chat`
    - **API Endpoint URL**：`http://{vllm_server_host}:{vllm_server_port}/v1`
    - **Model Name for API Endpoint**：`Qwen/Qwen1.5-7B-Chat`
    - **Completion Mode**：`Completion`

    ![](../../assets/deployment/dify-settings.png)

6. 如需创建测试聊天机器人，请进入 `Studio → Chatbot → Create from Blank`，然后选择 Chatbot 类型：

    ![](../../assets/deployment/dify-create-chatbot.png)

7. 点击刚创建的聊天机器人，打开聊天界面，即可开始与模型对话：

    ![](../../assets/deployment/dify-chat.png)