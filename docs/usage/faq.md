# 常见问题

> 问：如何通过 OpenAI API 在同一个端口上同时部署多个模型？

答：如果你指的是使用兼容 OpenAI 的服务器同时部署多个模型，目前暂不支持这样做。你可以选择运行多个服务器实例（每个实例服务一个不同的模型），然后通过额外的路由层将请求分发给对应的服务器。

---

> 问：做离线推理的 embedding（嵌入向量）任务推荐用哪个模型？

答：你可以尝试 [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 和 [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)  
更多模型可以参考[这里](../models/supported_models.md)。

通过提取隐藏状态，vLLM 可以自动将像 [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)、[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 这样的文本生成模型转换为 embedding 模型，  
但这些模型的效果通常不如专门为 embedding 任务训练的模型。

---

> 问：在 vLLM 中，同一个 prompt 的输出每次都一样吗？

答：不一定会一样。vLLM 并不保证输出 token 的对数概率（logprobs）是稳定的。  
由于 Torch 运算的数值不稳定性，或者当批处理方式发生变化时（如批量 Torch 运算的非确定性行为），logprobs 可能会有所波动。详情可参考 [数值精度相关说明](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。

在 vLLM 中，同样的请求可能因为以下原因被分在不同的批次：有其它并发请求、批次大小变化，或在 speculative decoding（猜测式解码）中批量扩展。  
这些批处理方式的变化，加上 Torch 运算本身的数值不稳定性，导致每一步的 logit/logprob 结果可能略有不同。  
这些微小的差异会逐步累积，最后可能导致采样出的 token 不同。一旦采样出不同的 token，后续输出也会进一步分歧。

## 应对策略

- 如果希望结果更稳定、方差更小，可以使用 `float32`，但会占用更多内存。
- 如果你在用 `bfloat16`，切换到 `float16` 可能也会有所帮助。
- 设置请求的随机种子，在 temperature > 0 时有助于提升生成结果的稳定性，但由于数值精度的差异，仍可能出现细微差别。