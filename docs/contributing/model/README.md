# 总览

!!! important
    许多解码器语言模型现在可以通过 [Transformers 后端](../../models/supported_models.md#transformers) 自动加载，无需在 vLLM 中单独实现。建议你先尝试使用 `vllm serve <model>` 命令！

vLLM 支持的模型是经过特别优化的 [PyTorch](https://pytorch.org/) 模型，能够利用多种[特性](../../features/README.md#compatibility-matrix)来提升性能表现。

将模型集成到 vLLM 的复杂度主要取决于模型本身的架构。
如果你的模型与 vLLM 已有的模型结构类似，集成过程会相对简单。
但如果你的模型包含新的算子（比如全新的注意力机制），集成难度则会更高。

请按照以下页面提供的步骤操作：

- [基础模型](basic.md)
- [模型注册](registration.md)
- [单元测试](tests.md)
- [多模态支持](multimodal.md)
- [语音转文本支持](transcription.md)

!!! tip
    如果你在将模型集成到 vLLM 的过程中遇到问题，欢迎随时在 [GitHub issue](https://github.com/vllm-project/vllm/issues) 提问，
    或加入我们的 [开发者 Slack 社区](https://slack.vllm.ai) 交流。
    我们非常乐意为你提供帮助！