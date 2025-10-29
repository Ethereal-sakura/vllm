# FP8 INC

vLLM 支持在 Intel® Gaudi® 2 和 Intel® Gaudi® 3 AI 加速器上，使用 Intel® Neural Compressor（INC）进行 FP8（8位浮点）权重和激活量化。目前，量化功能仅在 Llama 模型中经过验证。

Intel Gaudi 支持多种模块和函数的量化，包括但不限于 `Linear`、`KVCache`、`Matmul` 和 `Softmax`。更多详情请参考：[Supported Modules\\Supported Functions\\Custom Patched Modules](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-modules) 

!!! note
    在 Gaudi 加速器上运行量化模型时，vLLM 需要测量文件。FP8 模型的校准流程请参阅 [vLLM HPU extension](https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration/README.md) 包的说明。

!!! note
    `QUANT_CONFIG` 是一个环境变量，用于指定测量或量化 [JSON 配置文件](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-json-config-file-options) 的路径。测量配置文件在模型校准过程中用于收集测量数据，量化配置则在推理时使用。

## 使用 FP8 在线推理

完成模型校准并收集测量数据后，可以用以下命令在 vLLM 中运行 FP8 推理：

```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_measure_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --tensor_paralel_size 8
```

!!! tip
    如果只是原型开发或测试 FP8 模型，可以设置环境变量 `VLLM_SKIP_WARMUP=true` 来跳过预热阶段，这样可以节省大量时间。但在正式生产环境中不建议关闭预热，因为这会导致性能明显下降。

!!! tip
    使用 FP8 模型时，由于 FP8 操作编译时间较长，可能会遇到超时问题。可以通过以下环境变量调整超时设置：
    `VLLM_ENGINE_ITERATION_TIMEOUT_S` - 用于调整 vLLM 服务的超时时间，单位为秒。例如，600 表示 10 分钟。
    `VLLM_RPC_TIMEOUT` - 用于调整 OpenAI 兼容 API 的 RPC 协议超时，单位为微秒。例如，600000 表示 10 分钟。

## 使用 FP8 离线推理

要进行离线推理（需先完成模型校准流程）：

* 设置 "QUANT_CONFIG" 环境变量，指向包含 QUANTIZE 模式的 JSON 配置文件。
* 向 `LLM` 对象传递参数 `quantization=inc` 和 `kv_cache_dtype=fp8_inc`。
* 推理结束后，调用 model_executor 的 shutdown 方法。

```python
from vllm import LLM
llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc")
...
# 在需要的提示和采样参数上调用 llm.generate 方法。
...
llm.llm_engine.model_executor.shutdown()
```

## 模型权重加载的设备说明

未量化的权重会首先加载到 CPU，然后再进行量化，并传输到目标设备（HPU）用于模型执行。这样可以减少设备内存的占用，因为设备内只存储经过量化的权重。