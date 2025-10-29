# 多模态数据处理

为了在 vLLM 中实现多种优化，比如 [分块预填充](../configuration/optimization.md#chunked-prefill) 和 [前缀缓存](../features/automatic_prefix_caching.md)，我们通过 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor]，结合 HF（Hugging Face）processor 的输出，将占位特征 token（如 `<image>`）与多模态输入（如原始图片）进行对应。

以下是 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 的主要特性：

## 提示词更新检测

HF processor 的主要职责之一，就是用占位 token 更新提示词。例如：

- 在字符串开头插入特征占位 token（如 `<image><image>...<image>`，数量等于特征的大小）。
- 用特征占位 token（如 `<image><image>...<image>`，数量等于特征大小）替换已有的输入占位 token（如单张图片对应的 `<image>`）。

哪些 token 被更新，是建立占位特征 token 与多模态输入对应关系的关键。

在 vLLM 中，这些信息通过 [PromptUpdate][vllm.multimodal.processing.PromptUpdate] 在 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 方法中指定。我们可以通过检测更新后的 token 是否存在，自动判断 HF 是否对提示词进行了更新。

## 已分词的提示词输入

为了支持在独立进程中进行分词，我们允许与多模态数据一同传入输入 token 的 ID。

### 问题描述

通常，HF processor 主要遵循以下步骤：

1. 对文本进行分词
2. 处理多模态输入
3. 更新提示词

我们的要求是：

- 对于文本 + 多模态输入，要走完整的 1--3 步。
- 对于已分词文本 + 多模态输入，只需执行第 2--3 步。

如何在不重写 HF processor 的前提下实现这一点？可以尝试在不同输入上多次调用 HF processor：

- 对于文本 + 多模态输入，直接调用 HF processor。
- 对于已分词 + 多模态输入，仅针对多模态部分调用 processor。

HF processor 本身支持文本 + 多模态输入，但对于已分词 + 多模态输入，如果输入的占位 token 数量与多模态输入数量不一致，则会报错。

另外，已分词的文本没有经过 HF processor，因此我们需要自己执行第 3 步，确保输出 token 与多模态数据一一对应。

### 虚拟文本（Dummy text）

为了解决第一个问题，每个模型需要定义如何根据多模态输入数量生成虚拟文本，通过 [get_dummy_text][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text] 实现。这样我们就能生成与多模态输入对应的虚拟文本，并一同输入，获得处理后的多模态数据。

### 自动提示词更新

针对第二个问题，我们在 [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates] 中实现了与模型无关的代码，根据 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 输出的规范，自动用特征占位 token 更新提示词。

### 小结

借助虚拟文本和自动提示词更新机制，多模态 processor 最终可以同时支持文本和 token 形式的提示词，并配合多模态数据。详细逻辑可参考 [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main]。

## Processor 输出缓存

部分 HF processor（如 Qwen2-VL 的 processor）[处理速度非常慢](https://github.com/vllm-project/vllm/issues/9238) 。为缓解这个问题，我们会缓存 HF processor 的多模态输出，避免重复处理同一个多模态输入（如同一张图片）。

每当接收到新数据时，首先检查哪些数据已经在缓存中，哪些还没有。缺失的部分会批量送入 HF processor 处理，并加入缓存，然后与已缓存数据合并。

由于只处理缺失的多模态输入，输入的占位 token 数量与多模态输入数量不再一致，因此无法和文本提示词一起传给 HF processor。因此，我们需要将文本和多模态输入分开处理，并借助 [虚拟文本](#dummy-text) 避免 HF 报错。由于会跳过 HF 的提示词更新逻辑，后续还需应用 [自动提示词更新](#automatic-prompt-updating)，以保证输出 token 与多模态数据的对应关系正确。