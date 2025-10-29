# LangChain

vLLM 也可以通过 [LangChain](https://github.com/langchain-ai/langchain) 使用

要安装 LangChain，请运行以下命令

```bash
pip install langchain langchain_community -q
```

如果你想在单个或多个 GPU 上进行推理，可以使用 `langchain` 中的 `VLLM` 类。

??? code

    ```python
    from langchain_community.llms import VLLM

    llm = VLLM(
        model="mosaicml/mpt-7b",
        trust_remote_code=True,  # 对于 huggingface 模型这是必须的
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        # 分布式推理时使用
        # tensor_parallel_size=...,
    )

    print(llm("What is the capital of France ?"))
    ```

更多详细信息，请参考这个 [教程](https://python.langchain.com/docs/integrations/llms/vllm)