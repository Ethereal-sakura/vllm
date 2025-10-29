# vLLM V1

!!! announcement

    我们已经开始弃用 V0，详情请阅读 [RFC #18571](https://github.com/vllm-project/vllm/issues/18571)

V1 现已默认启用，适用于所有支持的使用场景，我们会逐步为计划支持的其它场景开放。欢迎在 [GitHub](https://github.com/vllm-project/vllm) 或 [vLLM Slack](https://inviter.co/vllm-slack) 分享你的反馈。

如需禁用 V1，请设置环境变量为：`VLLM_USE_V1=0`，并通过 GitHub issue 告诉我们你的原因！

## 为什么要推出 vLLM V1？

vLLM V0 成功支持了多种模型和硬件，但随着新特性的独立开发，系统变得越来越复杂。这种复杂度导致新功能集成变得困难，技术债务也逐渐积累，因此我们需要一个更加简洁统一的系统设计。

在 V0 成功经验的基础上，vLLM V1 保留了 V0 的稳定核心组件（如模型、GPU 内核和工具），同时对调度器、KV 缓存管理器、工作线程、采样器、API 服务器等核心系统进行了全面重构，打造更加一体化、易维护的框架，更好地支持持续发展和创新。

V1 的主要目标包括：

- 提供一个**简单、模块化、易于修改的代码库**
- 实现**高性能**，几乎不占用 CPU 资源
- **整合核心优化**，统一架构
- **零配置**，默认启用特性和优化，无需手动设置

升级到 V1 核心引擎后，特别是在长上下文场景下，性能有显著提升。性能基准测试稍后补充。

更多信息可阅读 vLLM V1 的博客文章 [vLLM V1: A Major Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（2025 年 1 月 27 日发布）。

本用户指南将持续更新，介绍 vLLM V1 带来的**重要变更与限制**。团队正积极将 V1 作为默认引擎，后续更多特性上线时本指南也会及时补充。

## 当前进展

每项功能对应以下几种状态之一：

- **🚀 优化完成**：几乎完全优化，无需进一步开发
- **🟢 可用**：功能完整，持续优化中
- **🚧 开发中**：正在积极开发
- **🟡 计划中**：未来计划开发（部分已有 PR/RFC）
- **🟠 延期**：V1 暂未支持，未来会重新引入
- **🔴 已弃用**：除非有强烈需求，否则不再支持

!!! note
    vLLM V1 采用统一调度器，通过简单字典（如 `{request_id: num_tokens}`）对每个请求动态分配固定的 Token 配额，将提示（prompt）和输出 Token 一视同仁，无需严格区分预填充和解码阶段。这使得分段预填充、前缀缓存、猜测式解码等特性得以支持。

V1 调度器支持多种调度策略，包括先进先出（FCFS）和基于优先级的调度（根据分配的优先级处理请求，FCFS 作为优先级相同时的判定标准），可通过 `--scheduling-policy` 参数配置。

### 硬件支持

| 硬件       | 状态                                         |
|------------|----------------------------------------------|
| **NVIDIA** | <nobr>🚀</nobr>                              |
| **AMD**    | <nobr>🟢</nobr>                              |
| **INTEL GPU**    | <nobr>🟢</nobr>                              |
| **TPU**    | <nobr>🟢</nobr>                              |
| **CPU**    | <nobr>🟢 (x86\_64/aarch64) 🟡 (MacOS) </nobr> |

!!! note

    更多硬件平台可以通过插件支持，例如：

    - [vllm-ascend](https://github.com/vllm-project/vllm-ascend)
    - [vllm-spyre](https://github.com/vllm-project/vllm-spyre)
    - [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
    - [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

    详情请查看对应项目的仓库。

### 模型支持

| 模型类型                     | 状态                                                                             |
|-----------------------------|----------------------------------------------------------------------------------|
| **仅解码模型**               | <nobr>🚀 优化完成</nobr>                                                         |
| **编码-解码模型**            | <nobr>🟢 仅支持 Whisper</nobr>                                                   |
| **Embedding 模型**           | <nobr>🟢 可用</nobr>                                                             |
| **Mamba 模型**               | <nobr>🟢 (Mamba-2), 🟢 (Mamba-1)</nobr>                                          |
| **多模态模型**               | <nobr>🟢 可用</nobr>                                                             |

下方有关于部分尚未支持或 V1 计划新增特性的模型的说明。

#### Embedding 模型

基础支持已实现，目前可用。

后续我们会考虑集成 [hidden states processor](https://github.com/vllm-project/vllm/issues/12249)，该方案基于 [global logits processor](https://github.com/vllm-project/vllm/pull/13360)，实现同一引擎实例下同时支持生成与 embedding 功能。

#### Mamba 模型

支持采用 selective state-space 机制（不同于标准 Transformer 注意力机制）的模型。
支持包含 Mamba-2 和 Mamba-1 层的模型（如 `Mamba2ForCausalLM`, `MambaForCausalLM`, `FalconMambaForCausalLM`）。

同时支持将 Mamba-2、Mamba-1 层与标准注意力层混合的混合模型（如 `BambaForCausalLM`、`Zamba2ForCausalLM`、`NemotronHForCausalLM`、`FalconH1ForCausalLM`、`GraniteMoeHybridForCausalLM`、`JambaForCausalLM`、`Plamo2ForCausalLM`）。

也支持采用其他机制的混合模型（如 `MiniMaxText01ForCausalLM`、`MiniMaxM1ForCausalLM`、`Lfm2ForCausalLM`）。

请注意，上述所有模型目前暂不支持前缀缓存（prefix caching）。

#### 编码-解码模型

目前仅支持 Whisper。对于需要在编码器和解码器间进行交叉注意力的其它模型（如 `BartForConditionalGeneration`、`MllamaForConditionalGeneration`），暂不支持。

### 功能特性

| 功能特性                                  | 状态                                                                             |
|-------------------------------------------|----------------------------------------------------------------------------------|
| **前缀缓存（Prefix Caching）**            | <nobr>🚀 优化完成</nobr>                                                          |
| **分段预填充（Chunked Prefill）**         | <nobr>🚀 优化完成</nobr>                                                          |
| **LoRA**                                 | <nobr>🚀 优化完成</nobr>                                                          |
| **Logprobs 计算**                        | <nobr>🟢 可用</nobr>                                                              |
| **FP8 KV 缓存**                          | <nobr>🟢 Hopper 设备上可用 (<https://github.com/vllm-project/vllm/pull/15191>)</nobr>|
| **Spec Decode**                          | <nobr>🚀 优化完成</nobr>                                                          |
| **带前缀缓存的 Prompt Logprobs**         | <nobr>🟡 计划中 ([RFC #13414](https://github.com/vllm-project/vllm/issues/13414))</nobr>|
| **结构化输出备用后端**                    | <nobr>🟢 可用</nobr>                                                              |
| **请求级结构化输出后端**                  | <nobr>🔴 已弃用</nobr>                                                            |
| **best_of**                              | <nobr>🔴 已弃用 ([RFC #13361](https://github.com/vllm-project/vllm/issues/13361))</nobr>|
| **单请求 Logits 处理器**                  | <nobr>🔴 已弃用 ([RFC #13360](https://github.com/vllm-project/vllm/pull/13360))</nobr> |
| **GPU <> CPU KV 缓存交换**               | <nobr>🔴 已弃用</nobr>                                                            |

!!! note

    vLLM V1 采用统一调度器，通过简单字典（如 `{request_id: num_tokens}`）对每个请求动态分配固定的 Token 配额，将提示和输出 Token 一视同仁，无需严格区分预填充和解码阶段。这使得分段预填充、前缀缓存、猜测式解码等特性得以支持。

#### Logprobs 语义变更

vLLM V1 支持 logprobs 以及 prompt logprobs，但与 V0 相比有一些重要语义差异：

##### Logprobs 计算方式

默认情况下，V1 会在模型原始输出后立刻返回 logprobs（即还未经过任何 logits 后处理，如温度缩放或惩罚项调整）。因此，最终返回的 logprobs 并不代表采样时实际使用的概率分布。

你可以通过设置 `--logprobs-mode` 参数调整这一行为。
共支持四种模式：`raw_logprobs`（默认）、`processed_logprobs`、`raw_logits`、`processed_logits`。
Raw 表示未经过任何 logits 处理器（如禁止词等）前的值；
Processed 表示经过所有处理器（包括温度、top_k/top_p 等）后的值。

##### 带前缀缓存的 Prompt Logprobs

logprobs 不会被缓存。如果请求需要 prompt logprobs，引擎会忽略前缀缓存，重新计算整个 prompt 的预填充以生成 logprobs。

#### 已弃用特性

由于 V1 的架构重大重构，部分旧特性被正式弃用。

##### 采样相关特性

- **best_of**：由于使用场景有限，该特性已弃用。详见 [RFC #13361](https://github.com/vllm-project/vllm/issues/13361)
- **单请求 Logits 处理器**：V0 支持为每个请求自定义 logits 处理函数，V1 已弃用该功能。后续将支持**全局 logits 处理器**，目前团队正在积极研发。详见 [RFC #13360](https://github.com/vllm-project/vllm/pull/13360)

##### KV 缓存相关特性

- **GPU <> CPU KV 缓存交换**：全新简化的核心架构下，V1 已无需通过 KV 缓存交换来处理请求抢占。

##### 结构化输出相关特性

- **请求级结构化输出后端**：已弃用，现已支持备用后端（outlines、guidance）和回退机制。