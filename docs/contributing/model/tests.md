# 单元测试

本页将介绍如何编写单元测试，以验证你的模型实现是否正确。

## 必须编写的测试

这些测试是你的 PR 合并到 vLLM 库所必需的。
如果没有这些测试，你的 PR 在持续集成（CI）中会失败。

### 模型加载

请在 [tests/models/registry.py](../../../tests/models/registry.py) 中为你的模型添加一个 HuggingFace 仓库的示例。
这样可以启用一个单元测试，使用虚拟权重加载你的模型，确保它能在 vLLM 中被正确初始化。

!!! important
    每个部分中的模型列表需要按照字母顺序维护。

!!! tip
    如果你的模型依赖于尚未发布的 HF Transformers 开发版，可以设置
    `min_transformers_version`，让 CI 跳过相关测试，直到模型正式发布。

## 可选测试

这些测试不是必须的，但通过这些测试可以进一步保证你的实现是正确的，并有助于避免后续出现功能回退等问题。

### 模型正确性

这些测试会将 vLLM 的模型输出与 [HF Transformers](https://github.com/huggingface/transformers) 进行对比。你可以在 [tests/models](../../../tests/models) 的子目录下添加新的测试用例。

#### 生成式模型

针对[生成式模型](../../models/generative_models.md)，在 [tests/models/utils.py](../../../tests/models/utils.py) 中定义了两类正确性测试：

- 精确一致性（`check_outputs_equal`）：vLLM 输出的文本应与 HF 输出的文本完全一致。
- Logprobs 相似性（`check_logprobs_close`）：vLLM 输出的 logprobs 应该出现在 HF 的 top-k logprobs 中，反之亦然。

#### 池化模型

对于[池化模型](../../models/pooling_models.md)，我们只需检查余弦相似度，具体定义见 [tests/models/utils.py](../../../tests/models/utils.py)。

### 多模态处理

#### 通用测试

将你的模型添加到 [tests/models/multimodal/processing/test_common.py](../../../tests/models/multimodal/processing/test_common.py) 中，可以验证以下输入组合是否输出一致：

- 文本 + 多模态数据
- token + 多模态数据
- 文本 + 已缓存的多模态数据
- token + 已缓存的多模态数据

#### 模型专属测试

你可以在 [tests/models/multimodal/processing](../../../tests/models/multimodal/processing) 下新建文件，编写只适用于你的模型的测试。

例如，如果你的模型的 HF processor 支持用户自定义的关键字参数，可以通过类似 [tests/models/multimodal/processing/test_phi3v.py](../../../tests/models/multimodal/processing/test_phi3v.py) 的方式，验证关键字参数是否被正确处理。