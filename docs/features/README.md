# 功能特性

## 兼容性矩阵

以下表格展示了部分硬件平台上，功能之间的互斥关系及支持情况。

各符号含义如下：

- ✅ = 完全兼容
- 🟠 = 部分兼容
- ❌ = 不兼容
- ❔ = 未知或待补充

!!! note
    带有 ❌ 或 🟠 并附带链接的格子，点击可查看对应功能与硬件组合的不支持追踪问题。

### 功能 x 功能

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 功能 | [CP](../configuration/optimization.md#chunked-prefill) | [APC](automatic_prefix_caching.md) | [LoRA](lora.md) | [SD](spec_decode.md) | CUDA graph | [pooling](../models/pooling_models.md) | <abbr title="Encoder-Decoder Models">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="Async Output Processing">async output</abbr> | multi-step | <abbr title="Multimodal Inputs">mm</abbr> | best-of | beam-search | [prompt-embeds](prompt_embeds.md) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [CP](../configuration/optimization.md#chunked-prefill) | ✅ | | | | | | | | | | | | | | |
| [APC](automatic_prefix_caching.md) | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| [SD](spec_decode.md) | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | |
| [pooling](../models/pooling_models.md) | 🟠\* | 🟠\* | ✅ | ❌ | ✅ | ✅ | | | | | | | | | |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ✅ | ✅ | ✅ | | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | | |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | | |
| multi-step | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | | |
| [mm](multimodal_inputs.md) | ✅ | ✅ | [🟠](https://github.com/vllm-project/vllm/pull/4194)<sup>^</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | | |
| best-of | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ✅ | ✅ | | |
| beam-search | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ❔ | ✅ | ✅ | |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❔ | ❔ | ❌ | ❔ | ❔ | ✅ |

\* 分块预填（chunked prefill）和前缀缓存（prefix caching）仅支持最后一个 token 的池化。  
<sup>^</sup> LoRA 仅适用于多模态模型的语言主干部分。

### 功能 x 硬件

| 功能                                                   | Volta               | Turing    | Ampere    | Ada    | Hopper     | CPU                | AMD    | TPU | Intel GPU |
|--------------------------------------------------------|---------------------|-----------|-----------|--------|------------|--------------------|--------|-----| ------------|
| [CP](../configuration/optimization.md#chunked-prefill) | [❌](https://github.com/vllm-project/vllm/issues/2729) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ | ✅        |
| [APC](automatic_prefix_caching.md)                     | [❌](https://github.com/vllm-project/vllm/issues/3687) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ | ✅        |
| [LoRA](lora.md)                                        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ | ✅        |
| [SD](spec_decode.md)                                   | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     | ❌ | [🟠](https://github.com/vllm-project/vllm/issues/26963)       |
| CUDA graph                                             | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     | ❌ | [❌](https://github.com/vllm-project/vllm/issues/26970)        |
| [pooling](../models/pooling_models.md)                 | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | ✅        |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr>    | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❌     | ❌ | ✅        |
| [mm](multimodal_inputs.md)                             | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | [🟠](https://github.com/vllm-project/vllm/issues/26965)       |
| <abbr title="Logprobs">logP</abbr>                     | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | ✅        |
| <abbr title="Prompt Logprobs">prmpt logP</abbr>        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | ✅        |
| <abbr title="Async Output Processing">async output</abbr> | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ❌     | ❌ | ✅        |
| multi-step                                             | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](https://github.com/vllm-project/vllm/issues/8477) | ✅     | ❌ | ✅        |
| best-of                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | ✅        |
| beam-search                                            | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ | ✅        |
| [prompt-embeds](prompt_embeds.md)                      | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❔     | [❌](https://github.com/vllm-project/vllm/issues/25097) | ✅       |
