---
hide:
  - navigation
  - toc
---

# 欢迎来到 vLLM

<figure markdown="span">
  ![](./assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM Light" class="logo-light" width="60%" }
  ![](./assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM Dark" class="logo-dark" width="60%" }
</figure>

<p style="text-align:center">
<strong>让大语言模型（LLM）服务变得简单、快速、低成本，人人都能用
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

vLLM 是一个高效且易用的大语言模型推理与服务库。

vLLM 最初由加州大学伯克利分校 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，如今已演变为社区驱动的开源项目，汇聚了学术界和工业界的贡献者。

如何开始使用 vLLM，取决于你的需求。如果你希望：

- 在 vLLM 上运行开源模型，建议先阅读 [快速上手指南](./getting_started/quickstart.md)
- 基于 vLLM 构建应用，建议查阅 [用户指南](./usage)
- 参与 vLLM 开发，建议参考 [开发者指南](./contributing)

关于 vLLM 的开发进展，可以查看：

- [开发路线图](https://roadmap.vllm.ai)
- [版本发布](https://github.com/vllm-project/vllm/releases)

vLLM 拥有极致的性能，包括：

- 行业领先的推理吞吐量
- 通过 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理注意力键值内存
- 支持持续批处理请求
- 基于 CUDA/HIP graph 的高速模型执行
- 量化支持： [GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、INT4、INT8 和 FP8
- 优化的 CUDA 内核，集成 FlashAttention 和 FlashInfer
- 支持 speculative decoding（推测解码）
- 支持分块预填充

vLLM 灵活且易用，特点包括：

- 可无缝集成 HuggingFace 流行模型
- 支持多种解码算法实现高吞吐量服务，如*并行采样*、*beam search*等
- 支持张量、流水线、数据和专家并行，实现分布式推理
- 支持流式输出
- 提供 OpenAI 兼容 API 服务端
- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU 以及 TPU，同时支持多种硬件插件，如 Intel Gaudi、IBM Spyre 和华为昇腾
- 支持前缀缓存（prefix caching）
- 支持多 LoRA（Low-Rank Adaptation）

想了解更多信息，欢迎访问以下内容：

- [vLLM 发布博客](https://vllm.ai)（介绍 PagedAttention）
- [vLLM 论文](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- [持续批处理如何提升 LLM 推理吞吐量 23 倍，降低 p50 延迟](https://www.anyscale.com/blog/continuous-batching-llm-inference)，作者 Cade Daniel 等
- [vLLM 线下交流会](community/meetups.md)
