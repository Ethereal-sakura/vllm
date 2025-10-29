# 概览

## 配置

vLLM 配置类的 API 文档。

- [vllm.config.ModelConfig][]
- [vllm.config.CacheConfig][]
- [vllm.config.LoadConfig][]
- [vllm.config.ParallelConfig][]
- [vllm.config.SchedulerConfig][]
- [vllm.config.DeviceConfig][]
- [vllm.config.SpeculativeConfig][]
- [vllm.config.LoRAConfig][]
- [vllm.config.MultiModalConfig][]
- [vllm.config.PoolerConfig][]
- [vllm.config.StructuredOutputsConfig][]
- [vllm.config.ObservabilityConfig][]
- [vllm.config.KVTransferConfig][]
- [vllm.config.CompilationConfig][]
- [vllm.config.VllmConfig][]

## 离线推理

LLM 类。

- [vllm.LLM][]

LLM 输入。

- [vllm.inputs.PromptType][]
- [vllm.inputs.TextPrompt][]
- [vllm.inputs.TokensPrompt][]

## vLLM 推理引擎

用于离线和在线推理的引擎类。

- [vllm.LLMEngine][]
- [vllm.AsyncLLMEngine][]

## 推理参数

vLLM API 的推理参数。

- [vllm.SamplingParams][]
- [vllm.PoolingParams][]

## 多模态

vLLM 通过 [vllm.multimodal][] 包提供了对多模态模型的实验性支持。

除了文本和 token 提示外，还可以通过 [vllm.inputs.PromptType][] 的 `multi_modal_data` 字段向[支持的模型](../models/supported_models.md#list-of-multimodal-language-models)传递多模态输入。

想要添加自定义的多模态模型？请参考[这里的说明](../contributing/model/multimodal.md)。

- [vllm.multimodal.MULTIMODAL_REGISTRY][]

### 输入接口

面向用户的输入数据。

- [vllm.multimodal.inputs.MultiModalDataDict][]

内部数据结构。

- [vllm.multimodal.inputs.PlaceholderRange][]
- [vllm.multimodal.inputs.NestedTensors][]
- [vllm.multimodal.inputs.MultiModalFieldElem][]
- [vllm.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.multimodal.inputs.MultiModalKwargsItems][]
- [vllm.multimodal.inputs.MultiModalKwargs][]
- [vllm.multimodal.inputs.MultiModalInputs][]

### 数据解析

- [vllm.multimodal.parse][]

### 数据处理

- [vllm.multimodal.processing][]

### 内存分析

- [vllm.multimodal.profiling][]

### 注册表

- [vllm.multimodal.registry][]

## 模型开发

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]