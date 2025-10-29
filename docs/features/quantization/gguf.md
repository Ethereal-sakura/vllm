# GGUF

!!! warning
    请注意，目前 vLLM 对 GGUF 的支持还处于高度实验阶段，且优化不足，可能与其他功能不兼容。目前，你可以使用 GGUF 来减少内存占用。如果遇到任何问题，请及时反馈给 vLLM 团队。

!!! warning
    目前，vllm 只支持加载单文件的 GGUF 模型。如果你拥有多文件的 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为单文件模型。

要在 vLLM 中运行 GGUF 模型，你可以从 [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) 下载本地 GGUF 模型，并通过以下命令加载：

```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# 建议使用基础模型的 tokenizer，可以避免耗时且容易出错的 tokenizer 转换过程。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

你还可以添加 `--tensor-parallel-size 2` 参数，在两张 GPU 上进行张量并行推理：

```bash
# 建议使用基础模型的 tokenizer，可以避免耗时且容易出错的 tokenizer 转换过程。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --tensor-parallel-size 2
```

!!! warning
    强烈建议使用基础模型的 tokenizer，而不是 GGUF 模型自带的 tokenizer。因为从 GGUF 模型转换 tokenizer 的过程非常耗时且不稳定，特别是对于词表较大的模型。

GGUF 假定 huggingface 能够将元数据转换为 config 文件。如果 huggingface 不支持你的模型，你可以手动创建 config，并通过 hf-config-path 参数传入。

```bash
# 如果 huggingface 不支持你的模型，可以手动指定一个兼容 huggingface 的 config 路径
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --hf-config-path Tinyllama/TInyLlama-1.1B-Chat-v1.0
```

你也可以通过 LLM 入口直接使用 GGUF 模型：

??? code

      ```python
      from vllm import LLM, SamplingParams

      # 本示例演示如何使用 chat 方法传递输入：
      conversation = [
         {
            "role": "system",
            "content": "你是一位乐于助人的助手",
         },
         {
            "role": "user",
            "content": "你好",
         },
         {
            "role": "assistant",
            "content": "你好！我有什么可以帮您的吗？",
         },
         {
            "role": "user",
            "content": "写一篇关于高等教育重要性的文章。",
         },
      ]

      # 创建采样参数对象。
      sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

      # 创建 LLM 实例。
      llm = LLM(
         model="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
         tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      )
      # 根据输入生成文本。输出为 RequestOutput 对象列表，
      # 包含原始输入、生成结果及其他信息。
      outputs = llm.chat(conversation, sampling_params)

      # 打印输出结果。
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
      ```