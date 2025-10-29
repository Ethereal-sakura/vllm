# 量化（Quantization）

量化是一种用模型精度换取更小内存占用的技术，这样可以让大型模型在更多类型的设备上运行。

目录：

- [AutoAWQ](auto_awq.md)
- [AutoRound](auto_round.md)
- [BitsAndBytes](bnb.md)
- [BitBLAS](bitblas.md)
- [GGUF](gguf.md)
- [GPTQModel](gptqmodel.md)
- [INC](inc.md)
- [INT4 W4A16](int4.md)
- [INT8 W8A8](int8.md)
- [FP8 W8A8](fp8.md)
- [NVIDIA TensorRT Model Optimizer](modelopt.md)
- [AMD Quark](quark.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)

## 支持的硬件平台

下表展示了在 vLLM 中，不同量化实现与各类硬件平台的兼容情况：

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

| Implementation        | Volta   | Turing   | Ampere   | Ada   | Hopper   | AMD GPU   | Intel GPU   | Intel Gaudi | x86 CPU   | Google TPU   |
|-----------------------|---------|----------|----------|-------|----------|-----------|-------------|-------------|-----------|--------------|
| AWQ                   | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ❌         | ✅︎        | ❌           |
| GPTQ                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ✅︎          | ❌         | ✅︎        | ❌           |
| Marlin (GPTQ/AWQ/FP8) | ❌      | ❌       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        | ❌           |
| INT8 (W8A8)           | ❌      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ✅︎        | ✅︎           |
| FP8 (W8A8)            | ❌      | ❌       | ❌       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌         | ❌        | ❌           |
| BitBLAS               | ✅︎      | ✅       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        | ❌           |
| BitBLAS (GPTQ)        | ❌      | ❌       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        | ❌           |
| bitsandbytes          | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        | ❌           |
| DeepSpeedFP           | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ❌         | ❌          | ❌         | ❌        | ❌           |
| GGUF                  | ✅︎      | ✅︎       | ✅︎       | ✅︎    | ✅︎       | ✅︎         | ❌          | ❌         | ❌        | ❌           |
| INC (W8A8)            | ❌      | ❌       | ❌       | ❌    | ❌       | ❌         | ❌          | ✅︎         | ❌        | ❌           |

- Volta 指的是 SM 7.0，Turing 是 SM 7.5，Ampere 为 SM 8.0/8.6，Ada 为 SM 8.9，Hopper 为 SM 9.0
- ✅︎ 表示该量化方法在对应硬件上受支持
- ❌ 表示该量化方法在对应硬件上不受支持

!!! note
    此兼容性表格会随着 vLLM 持续更新和对新硬件及量化方法的支持而变化。

    获取最新的硬件支持与量化方法信息，请参考 [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) 或联系 vLLM 开发团队