# 检索增强生成（Retrieval-Augmented Generation）

[检索增强生成（Retrieval-augmented generation，简称 RAG）](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) 是一种让生成式人工智能（Generative AI，Gen AI）模型能够检索并融合新信息的技术。它通过调整与大语言模型（Large Language Model，LLM）的交互方式，使模型在回答用户提问时，能够参考一组指定的文档，将这些文档的信息与模型已有的训练数据结合起来补充答案。这样，LLM 就可以利用特定领域的信息或最新的数据。例如，可以让聊天机器人访问公司内部数据，或基于权威来源生成回复。

目前支持的集成方式有：

- vLLM + [langchain](https://github.com/langchain-ai/langchain) + [milvus](https://github.com/milvus-io/milvus)
- vLLM + [llamaindex](https://github.com/run-llama/llama_index) + [milvus](https://github.com/milvus-io/milvus)

## vLLM + langchain

### 环境准备

先配置好 vLLM 和 langchain 的环境：

```bash
pip install -U vllm \
            langchain_milvus langchain_openai \
            langchain_community beautifulsoup4 \
            langchain-text-splitters
```

### 部署步骤

1. 启动支持 embedding 的 vLLM 服务，例如：

    ```bash
    # 启动 embedding 服务（端口 8000）
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

2. 启动支持对话生成的 vLLM 服务，例如：

    ```bash
    # 启动聊天服务（端口 8001）
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

3. 使用脚本：[examples/online_serving/retrieval_augmented_generation_with_langchain.py](../../../examples/online_serving/retrieval_augmented_generation_with_langchain.py)

4. 运行脚本

    ```bash
    python retrieval_augmented_generation_with_langchain.py
    ```

## vLLM + llamaindex

### 环境准备

先配置好 vLLM 和 llamaindex 的环境：

```bash
pip install vllm \
            llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \
```

### 部署步骤

1. 启动支持 embedding 的 vLLM 服务，例如：

    ```bash
    # 启动 embedding 服务（端口 8000）
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

2. 启动支持对话生成的 vLLM 服务，例如：

    ```bash
    # 启动聊天服务（端口 8001）
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

3. 使用脚本：[examples/online_serving/retrieval_augmented_generation_with_llamaindex.py](../../../examples/online_serving/retrieval_augmented_generation_with_llamaindex.py)

4. 运行脚本：

    ```bash
    python retrieval_augmented_generation_with_llamaindex.py
    ```