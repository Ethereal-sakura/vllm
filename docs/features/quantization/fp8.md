# FP8 W8A8

vLLM 支持在 Nvidia H100 和 AMD MI300x 等 GPU 上利用硬件加速进行 FP8（8 位浮点数）权重与激活量的量化。当前，W8A8 量化仅官方支持 Hopper 和 Ada Lovelace 架构的 GPU。Ampere 架构则支持 W8A16（仅权重量化 FP8），通过 Marlin 内核实现。采用 FP8 对模型进行量化，可以将模型内存占用减少至一半，并将吞吐量提升最高 1.6 倍，同时对准确性影响极小。

欢迎访问 Hugging Face 上的 [适用于 vLLM 的主流 LLM FP8 量化模型集合](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127)，可直接下载使用。

硬件支持的 FP8 类型通常有两种不同的表示方法，适用于不同场景：

- **E4M3**：由 1 位符号位、4 位指数位和 3 位尾数位组成。可表示数值范围为 +/-448，以及 `nan`。
- **E5M2**：由 1 位符号位、5 位指数位和 2 位尾数位组成。可表示数值范围为 +/-57344，支持 +/- `inf` 和 `nan`。动态范围变大但精度有所降低。

!!! note
    FP8 运算仅在 NVIDIA 计算能力大于 8.9 的 GPU（如 Ada Lovelace, Hopper）上支持。
    FP8 模型在计算能力大于 8.0 的 GPU（Ampere）上以仅权重 W8A16 模式运行，使用 FP8 Marlin。

## 安装

要在 vLLM 中高效生成 FP8 量化模型，需要先安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```bash
pip install llmcompressor
```

## 量化流程

整个量化流程分为三步：

1. 加载模型
2. 执行量化
3. 在 vLLM 中评估准确率

### 1. 加载模型

使用标准的 `transformers` AutoModel 类加载模型和分词器：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. 执行量化

对于 FP8 量化，可以使用简单的 RTN 量化方法实现高准确率。推荐对所有 `Linear` 层采用 `FP8_DYNAMIC` 方案，该方案包括：

- 权重采用静态、按通道量化
- 激活量采用动态、按 token 量化

由于简单的 RTN 权重量化无需数据，激活量在推理时动态量化，因此无需校准数据。

??? code

    ```python
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # 配置简单的 PTQ 量化
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    # 应用量化算法
    oneshot(model=model, recipe=recipe)

    # 保存模型：Meta-Llama-3-8B-Instruct-FP8-Dynamic
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

### 3. 评估准确率

安装 `vllm` 和 `lm-evaluation-harness` 用于评测：

```bash
pip install vllm git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api]
```

在 `vllm` 中加载并运行模型：

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
print(result[0].outputs[0].text)
```

用 `lm_eval` 评估准确率（例如在 `gsm8k` 任务上抽取 250 个样本）：

!!! note
    量化模型对 `bos` token 是否存在非常敏感。`lm_eval` 默认不添加 `bos` token，请在评测时务必加上 `add_bos_token=True` 参数。

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

结果示例：

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## 常见问题与支持

如遇到问题或有功能需求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) 的 GitHub 仓库提交 Issue。

## 在线动态量化

无需校准数据，vLLM 可对原始精度的 BF16/FP16 模型进行在线 FP8 动态量化。只需在命令行中加入 `--quantization="fp8"`，或在 LLM 构造器中设置 `quantization="fp8"` 即可启用。

在这种模式下，所有 Linear 层（最终的 `lm_head` 除外）权重会量化到 FP8_E4M3 精度，采用每个张量一个缩放因子。激活量在每次前向计算时动态统计最小/最大值，生成相应缩放因子，实现高准确率。此模式下推理延迟提升有限。

```python
from vllm import LLM

llm = LLM("facebook/opt-125m", quantization="fp8")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```

!!! warning
    当前模型会先以原始精度加载，再量化到 8 位，因此需要足够内存来加载完整模型。
