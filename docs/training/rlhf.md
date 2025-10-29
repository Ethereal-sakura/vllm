# 基于人类反馈的强化学习

基于人类反馈的强化学习（Reinforcement Learning from Human Feedback，简称 RLHF）是一种利用人类生成的偏好数据对语言模型进行微调的方法，旨在让模型输出更符合期望的行为。vLLM 可以用于生成 RLHF 过程中的补全文本。

以下这些开源的强化学习（RL）库会用 vLLM 来实现高效的 rollout（按字母顺序排列，非完整列表）：

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

如果你不想直接使用现有的 RL 库，可以参考以下基础示例快速上手：

- [训练和推理过程分布在不同的 GPU 上（灵感来自 OpenRLHF）](../examples/offline_inference/rlhf.md)
- [训练和推理过程共用同一块 GPU，基于 Ray 实现](../examples/offline_inference/rlhf_colocate.md)
- [使用 vLLM 进行 RLHF 的实用工具](../examples/offline_inference/rlhf_utils.md)

下面这些示例 notebook 展示了如何结合 vLLM 实现 GRPO：

- [在 TRL 中结合 GRPO 与 vLLM 实现高效在线训练](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [使用 Unsloth + vLLM 实现 Qwen-3 4B 的 GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
