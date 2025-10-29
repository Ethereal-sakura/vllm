# AutoRound

[AutoRound](https://github.com/intel/auto-round) 是英特尔推出的先进量化算法，能够高效地对大语言模型进行 **INT2、INT3、INT4 和 INT8** 量化，在模型精度和部署性能之间实现了最佳平衡。

AutoRound 采用权重量化（weight-only quantization）技术，面向 Transformer 架构模型，不仅大幅节省内存，还能加快推理速度，同时保持接近原始模型的准确率。它支持多种硬件平台，包括 **CPU、英特尔 GPU、HPU 以及支持 CUDA 的设备**。

详细内容请参考 [AutoRound 指南](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md) 

主要特性：

✅ 支持 **AutoRound、AutoAWQ、AutoGPTQ 和 GGUF**

✅ 支持 **10+ 种视觉-语言模型（VLMs）**

✅ 支持 **逐层混合比特量化**，实现更精细的控制

✅ 提供 **RTN（Round-To-Nearest）模式**，可快速量化，仅带来微小精度损失

✅ 多种量化策略选择：best、base 和 light

✅ 提供高级工具，如即时打包，并支持 **10+ 种后端**

## 安装方式

```bash
uv pip install auto-round
```

## 对模型进行量化

如需量化视觉-语言模型（VLMs），命令行中请将 `auto-round` 替换为 `auto-round-mllm`，API 调用中使用 `AutoRoundMLLM`。

### 命令行用法

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --format "gguf:q4_k_m" \
    --output_dir ./tmp_autoround
```

### API 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

# 最高精度，速度会慢 4-5 倍，low_gpu_mem_usage 可节省约 20G 显存，但速度会下降约 30%
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 速度提升 2-3 倍，W4G128 配置下精度略有下降
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format 可选：'auto_round'(默认)、'auto_gptq'、'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## 使用 vLLM 运行量化后的模型

以下是使用 vLLM 加载 auto-round 格式模型的示例代码：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
llm = LLM(model=model_name)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 致谢

特别感谢 AutoGPTQ、AutoAWQ、GPTQModel、Triton、Marlin 以及 ExLLaMAV2 等开源低精度库，为 AutoRound 提供了低精度 CUDA 内核支持。
