# GPTQModel

要创建新的 4 位或 8 位 GPTQ 量化模型，可以使用 ModelCloud.AI 提供的 [GPTQModel](https://github.com/ModelCloud/GPTQModel)。

量化（Quantization）会将模型的精度从 BF16/FP16 (16 位) 降低到 INT4 (4 位) 或 INT8 (8 位)，这样不仅能显著减少模型的整体内存占用，还能同时提升推理性能。

经过 GPTQModel 量化的模型可以利用 `Marlin` 和 `Machete` 两款 vLLM 定制内核，在 Ampere（A100+）和 Hopper（H100+）系列 Nvidia GPU 上最大化批量处理速率（transactions-per-second，简称 tps）和单 token 延迟性能。这两款内核由 vLLM 和 NeuralMagic（现已归属 Redhat）深度优化，使量化后的 GPTQ 模型在推理速度上达到世界级水平。

GPTQModel 是全球少数支持 `Dynamic`（动态）分模块量化（per-module quantization）的工具之一，允许对大型语言模型（llm）中的不同层或模块定制化量化参数，实现进一步优化。`Dynamic` 量化已与 vLLM 深度集成，并且由 ModelCloud.AI 团队提供技术支持。更多详情和高级功能请参阅 [GPTQModel readme](https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override)。

## 安装

你可以通过安装 [GPTQModel](https://github.com/ModelCloud/GPTQModel) 自己对模型进行量化，也可以直接在 Huggingface 平台选用已有的 [5000+ GPTQ 模型](https://huggingface.co/models?search=gptq)。

```bash
pip install -U gptqmodel --no-build-isolation -v
```

## 模型量化

安装好 GPTQModel 后，即可开始模型量化。详细操作方法请参考 [GPTQModel readme](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization)。

以下是对 `meta-llama/Llama-3.2-1B-Instruct` 进行量化的示例：

??? code

    ```python
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(1024))["text"]

    quant_config = QuantizeConfig(bits=4, group_size=128)

    model = GPTQModel.load(model_id, quant_config)

    // 可根据显卡和显存的实际情况增大 batch_size，以加快量化速度
    model.quantize(calibration_dataset, batch_size=2)

    model.save(quant_path)
    ```

## 使用 vLLM 运行量化模型

如果你想用 vLLM 运行 GPTQModel 量化后的模型，可以参考下列命令，以 [DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2) 为例：

```bash
python examples/offline_inference/llm_engine_example.py \
    --model ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

## 结合 vLLM Python API 使用 GPTQModel

GPTQModel 量化模型也可通过 LLM 的入口直接支持：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 示例 prompt 列表
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 创建采样参数对象
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

    # 创建 LLM 对象
    llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

    # 根据 prompts 生成文本。输出会是包含 prompt、生成文本及其它信息的 RequestOutput 对象列表
    outputs = llm.generate(prompts, sampling_params)

    # 输出生成内容
    print("-"*50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-"*50)
    ```