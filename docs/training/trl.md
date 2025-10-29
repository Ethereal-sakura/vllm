# Transformers 强化学习

[Transformers Reinforcement Learning](https://huggingface.co/docs/trl)（TRL）是一个全栈库，提供了一套用于训练 transformer 语言模型的工具，支持如有监督微调（Supervised Fine-Tuning，SFT）、群体相对策略优化（Group Relative Policy Optimization，GRPO）、直接偏好优化（Direct Preference Optimization，DPO）、奖励建模（Reward Modeling）等方法。该库与 🤗 transformers 集成。

像 GRPO 或在线 DPO 这样的在线方法，需要模型生成补全内容。你可以使用 vLLM 来生成这些补全！

更多信息请参考 TRL 文档中的 [vLLM 集成指南](https://huggingface.co/docs/trl/main/en/vllm_integration)

目前，TRL 支持以下与 vLLM 集成的在线训练器：

- [GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [在线 DPO](https://huggingface.co/docs/trl/main/en/online_dpo_trainer)
- [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)
- [Nash-MD](https://huggingface.co/docs/trl/main/en/nash_md_trainer)
- [XPO](https://huggingface.co/docs/trl/main/en/xpo_trainer)

要在 TRL 中启用 vLLM，只需在训练器配置中将 `use_vllm` 标志设置为 `True`。

## 训练中使用 vLLM 的两种模式

TRL 支持在训练过程中集成 vLLM 的**两种模式**：**服务模式（server mode）**和**共置模式（colocate mode）**。你可以通过 `vllm_mode` 参数来控制 vLLM 的运行方式。

### 服务模式（server mode）

在**服务模式**下，vLLM 会作为独立进程运行在专用 GPU 上，并通过 HTTP 请求与训练器通信。如果你有专门用于推理的 GPU，这种方式非常适合，可以将生成任务和训练任务分离，保证性能稳定，也更容易扩展。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # 默认值，可省略
)
```

### 共置模式（colocate mode）

在**共置模式**下，vLLM 会在训练进程内部运行，并与训练模型共享 GPU 显存。这样无需启动独立服务，可以提升 GPU 的利用率，但可能会导致训练 GPU 的显存竞争。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

部分训练器还支持**vLLM 休眠模式（vLLM sleep mode）**，可以在训练时将参数和缓存数据卸载到 GPU 内存，有效降低显存占用。详细信息请参阅 [内存优化文档](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#vllm-sleep-mode)

!!! info
    更多详细的配置选项和参数说明，请参考你所使用训练器的官方文档。