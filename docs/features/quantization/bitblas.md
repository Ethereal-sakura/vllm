# BitBLAS

vLLM 现在支持 [BitBLAS](https://github.com/microsoft/BitBLAS)，可以实现更高效、更灵活的模型推理。与其他量化框架相比，BitBLAS 提供了更多的精度组合选择。

!!! note
    请确保你的硬件支持所选的 `dtype`（`torch.bfloat16` 或 `torch.float16`）。
    目前大多数新款 NVIDIA GPU 支持 `float16`，而 `bfloat16` 主要在 Ampere 或 Hopper 等新架构上更常见。
    更多信息请参考 [支持的硬件](README.md#supported-hardware)

下面是如何在 vLLM 中使用 BitBLAS 的步骤。

```bash
pip install bitblas>=0.1.0
```

vLLM 会读取模型的配置文件，并支持已经预量化的 checkpoint。

你可以在以下位置找到预量化的模型：

- [Hugging Face (BitBLAS)](https://huggingface.co/models?search=bitblas)
- [Hugging Face (GPTQ)](https://huggingface.co/models?search=gptq)

通常，这些模型仓库中会有一个 `quantize_config.json` 文件，其中包含了 `quantization_config` 配置。

## 读取 bitblas 格式的 checkpoint

```python
from vllm import LLM
import torch

# "hxbgsyxh/llama-13b-4bit-g-1-bitblas" 是一个已预量化的 checkpoint。
model_id = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitblas",
)
```

## 读取 gptq 格式的 checkpoint

??? code

    ```python
    from vllm import LLM
    import torch

    # "hxbgsyxh/llama-13b-4bit-g-1" 是一个已预量化的 checkpoint。
    model_id = "hxbgsyxh/llama-13b-4bit-g-1"
    llm = LLM(
        model=model_id,
        dtype=torch.float16,
        trust_remote_code=True,
        quantization="bitblas",
        max_model_len=1024,
    )
    ```