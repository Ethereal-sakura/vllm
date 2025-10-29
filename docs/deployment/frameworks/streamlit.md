# Streamlit

[Streamlit](https://github.com/streamlit/streamlit) 可以让你在几分钟内把 Python 脚本变成交互式网页应用，而不是花费数周时间。你可以用它来构建数据仪表盘、生成报告，或者创建聊天应用。

它可以非常方便地和 vLLM 作为后端 API 服务集成，通过 API 调用实现强大的大语言模型（LLM）推理能力。

## 前置条件

请先配置好 vLLM 运行环境，并安装所有必需的依赖包：

```bash
pip install vllm streamlit openai
```

## 部署步骤

1. 启动 vLLM 服务端，并加载一个支持聊天补全的模型，例如：

    ```bash
    vllm serve Qwen/Qwen1.5-0.5B-Chat
    ```

2. 使用示例脚本：[examples/online_serving/streamlit_openai_chatbot_webserver.py](../../../examples/online_serving/streamlit_openai_chatbot_webserver.py)

3. 启动 Streamlit 网页界面，开始对话：

    ```bash
    streamlit run streamlit_openai_chatbot_webserver.py

    # 或者指定 VLLM_API_BASE 或 VLLM_API_KEY
    VLLM_API_BASE="http://vllm-server-host:vllm-server-port/v1" \
        streamlit run streamlit_openai_chatbot_webserver.py

    # 以调试模式启动，可以查看更多详细信息
    streamlit run streamlit_openai_chatbot_webserver.py --logger.level=debug
    ```

    ![在 Streamlit 中和 vLLM 助手对话](../../assets/deployment/streamlit-chat.png)
