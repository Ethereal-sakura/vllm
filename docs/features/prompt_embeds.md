# Prompt Embedding 输入

本页面将教你如何向 vLLM 传递 prompt embedding（提示嵌入）输入。

## 什么是 prompt embeddings（提示嵌入）？

在大型语言模型（Large Language Model，LLM）中，文本数据的传统处理流程是：先将文本转换为 token ids（令牌 ID，借助分词器 tokenizer），再由 token ids 变为 prompt embeddings（提示嵌入）。对于传统的仅解码模型（decoder-only model，例如 meta-llama/Llama-3.1-8B-Instruct），将 token ids 转换为 prompt embeddings 的过程，是通过查找一个学习得到的嵌入矩阵完成的。不过，模型实际上并不局限于只处理那些与其 token 字典对应的嵌入。

## 离线推理

如果你需要输入多模态数据，可以按照 [vllm.inputs.EmbedsPrompt][] 中的结构来操作：

- `prompt_embeds`：这是一个 torch tensor，表示一系列提示/令牌嵌入。其形状为 (sequence_length, hidden_size)，其中 sequence_length 表示嵌入的令牌数量，hidden_size 是模型的隐藏层维度（即嵌入维度）。

### Hugging Face Transformers 输入

你可以将 Hugging Face Transformers 模型生成的 prompt embeddings 填入提示嵌入字典的 `'prompt_embeds'` 字段，参考以下示例：

[examples/offline_inference/prompt_embed_inference.py](../../examples/offline_inference/prompt_embed_inference.py)

## 在线服务

我们的 OpenAI 兼容服务端支持通过 [Completions API](https://platform.openai.com/docs/api-reference/completions) 接收 prompt embeddings 输入。你只需在 JSON 包中添加新的 `'prompt_embeds'` 键，并在运行 `vllm serve` 时启用 `--enable-prompt-embeds` 标志即可。

如果在同一个请求中同时提供 `'prompt_embeds'` 和 `'prompt'`，则提示嵌入会始终优先处理和返回。

提示嵌入应以 base64 编码的 torch tensor 方式传递。

!!! warning
    如果嵌入的形状不正确，vLLM 引擎可能会崩溃。
    请只对可信用户启用此功能！

### 通过 OpenAI 客户端输入 Transformers 数据

首先，启动 OpenAI 兼容服务端：

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --runner generate \
  --max-model-len 4096 --enable-prompt-embeds
```

然后，可以像下面这样使用 OpenAI 客户端：

[examples/online_serving/prompt_embed_inference_with_openai_client.py](../../examples/online_serving/prompt_embed_inference_with_openai_client.py)
