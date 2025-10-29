# BitsAndBytes

vLLM 现已支持 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)，让模型推理更加高效。  
BitsAndBytes 可以对模型进行量化（quantization），从而降低内存消耗、提升性能，同时基本不会影响模型的准确率。  
与其他量化方法相比，BitsAndBytes 无需使用输入数据对量化后的模型进行校准。

以下是如何在 vLLM 中使用 BitsAndBytes 的步骤。

```bash
pip install bitsandbytes>=0.46.1
```

vLLM 会自动读取模型的配置文件，支持动态量化（in-flight quantization）和已量化的 checkpoint 两种方式。

你可以在 [Hugging Face](https://huggingface.co/models?search=bitsandbytes) 上找到已经用 bitsandbytes 量化过的模型。  
这些模型仓库通常会包含一个 config.json 文件，其中包含 quantization_config 配置项。

## 读取已量化的 checkpoint

对于已经量化好的 checkpoint，vLLM 会自动从配置文件中识别出量化方式，无需手动指定 quantization 相关参数。

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit 是一个已经量化好的 checkpoint。
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

## 动态量化：以 4bit 方式加载

如果你希望在加载模型时使用 BitsAndBytes 进行 4bit 动态量化，需要显式指定 quantization 参数。

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
)
```

## OpenAI 兼容服务

如果使用 OpenAI 兼容服务器，想启用 4bit 动态量化，可以在模型参数中添加：

```bash
--quantization bitsandbytes
```
