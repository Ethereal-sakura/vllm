# AMD Quark

量化（Quantization）能够有效减少内存和带宽的使用，加速计算，提升吞吐量，同时只带来极小的精度损失。vLLM 支持借助 [Quark](https://quark.docs.amd.com/latest/) —— 一款灵活且强大的量化工具包，在 AMD GPU 上生成高性能的量化模型。Quark 针对大语言模型提供了专业的支持，包括权重、激活值和 kv-cache 的量化，以及前沿的量化算法，如 AWQ、GPTQ、Rotation 和 SmoothQuant。

## Quark 安装

在开始量化模型之前，需要先安装 Quark。可以使用 pip 安装最新版 Quark：

```bash
pip install amd-quark
```

更多安装细节，可以参考 [Quark 安装指南](https://quark.docs.amd.com/latest/install.html)。

此外，还需要安装 `vllm` 和 `lm-evaluation-harness` 以便进行评测：

```bash
pip install vllm git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api]
```

## 量化流程

安装好 Quark 后，我们以一个例子来演示如何使用 Quark。整个量化流程主要分为以下 5 个步骤：

1. 加载模型
2. 准备校准（calibration）数据加载器
3. 设置量化配置
4. 执行量化并导出模型
5. 在 vLLM 中评估

### 1. 加载模型

Quark 通过 [Transformers](https://huggingface.co/docs/transformers/en/index) 获取模型和分词器。

??? code

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
    MAX_SEQ_LEN = 512

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
    tokenizer.pad_token = tokenizer.eos_token
    ```

### 2. 准备校准数据加载器

Quark 使用 [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 来加载校准数据。关于如何高效使用校准数据集，可以参考 [添加校准数据集](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html)。

??? code

    ```python
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    BATCH_SIZE = 1
    NUM_CALIBRATION_DATA = 512

    # 加载数据集并获取校准数据。
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    text_data = dataset["text"][:NUM_CALIBRATION_DATA]

    tokenized_outputs = tokenizer(
        text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    calib_dataloader = DataLoader(
        tokenized_outputs['input_ids'],
        batch_size=BATCH_SIZE,
        drop_last=True,
    )
    ```

### 3. 设置量化配置

我们需要设置量化配置。更多配置细节可以参考 [quark 配置指南](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html)。本例中，我们对权重、激活和 kv-cache 都采用 FP8 每张量（per-tensor）量化，所用算法为 AutoSmoothQuant。

!!! note
    注意：量化算法需要一个 JSON 配置文件，配置文件位于 [Quark Pytorch 示例](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html)中的 `examples/torch/language_modeling/llm_ptq/models` 目录。例如，Llama 的 AutoSmoothQuant 配置文件是 `examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json`。

??? code

    ```python
    from quark.torch.quantization import (Config, QuantizationConfig,
                                        FP8E4M3PerTensorSpec,
                                        load_quant_algo_config_from_file)

    # 定义 fp8/每张量/静态 的量化规格。
    FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(
        observer_method="min_max",
        is_dynamic=False,
    ).to_quantization_spec()

    # 定义全局量化配置，输入张量和权重均采用 FP8_PER_TENSOR_SPEC。
    global_quant_config = QuantizationConfig(
        input_tensors=FP8_PER_TENSOR_SPEC,
        weight=FP8_PER_TENSOR_SPEC,
    )

    # 针对 kv-cache 层的量化配置，输出张量同样采用 FP8_PER_TENSOR_SPEC。
    KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
    kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
    kv_cache_quant_config = {
        name: QuantizationConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=KV_CACHE_SPEC,
        )
        for name in kv_cache_layer_names_for_llama
    }
    layer_quant_config = kv_cache_quant_config.copy()

    # 通过配置文件定义算法配置。
    LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = "examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json"
    algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        exclude=EXCLUDE_LAYERS,
        algo_config=algo_config,
    )
    ```

### 4. 执行量化并导出模型

接下来可以执行量化。量化完成后，需先“冻结”量化后的模型再导出。注意导出时需采用 HuggingFace 的 `safetensors` 格式，更多细节可以参考 [HuggingFace 格式导出说明](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html)。

??? code

    ```python
    import torch
    from quark.torch import ModelQuantizer, ModelExporter
    from quark.torch.export import ExporterConfig, JsonExporterConfig

    # 应用量化过程。
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    # 冻结量化后的模型，便于导出。
    freezed_model = quantizer.freeze(model)

    # 定义导出配置。
    LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
    export_config = ExporterConfig(json_export_config=JsonExporterConfig())
    export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP

    # 模型名举例：Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant
    EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
    exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
    with torch.no_grad():
        exporter.export_safetensors_model(
            freezed_model,
            quant_config=quant_config,
            tokenizer=tokenizer,
        )
    ```

### 5. 在 vLLM 中评估

现在，你可以通过 LLM 的入口，直接加载并运行 Quark 量化后的模型：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 示例 prompt
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # 创建采样参数对象
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 创建 LLM 实例
    llm = LLM(
        model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
        kv_cache_dtype="fp8",
        quantization="quark",
    )
    # 基于 prompts 生成文本，输出为 RequestOutput 对象列表，包含 prompt、生成文本及其它信息
    outputs = llm.generate(prompts, sampling_params)
    # 打印输出结果
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    ```

你也可以用 `lm_eval` 工具评测准确率：

```bash
lm_eval --model vllm \
  --model_args pretrained=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```

## Quark 量化脚本

除了上面 Python API 的例子外，Quark 还提供了一个更为便捷的 [量化脚本](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)，可以快速完成大语言模型的量化任务。该脚本支持多种量化方案与优化算法，可以直接导出量化模型并支持评测。使用脚本，上述示例可以简化为：

```bash
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```

## 使用 OCP MX (MXFP4, MXFP6) 模型

vLLM 支持加载通过 AMD Quark 离线量化得到的 MXFP4 和 MXFP6 模型，并符合 [Open Compute Project (OCP) 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

目前，该方案只支持对激活进行动态量化。

安装最新版 AMD Quark 后，示例用法如下：

```bash
vllm serve fxmarty/qwen_1.5-moe-a2.7b-mxfp4 --tensor-parallel-size 1
# 或者，对于激活使用 fp6、权重使用 fp4 的模型：
vllm serve fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3 --tensor-parallel-size 1
```

在不原生支持 OCP MX 操作的设备（如 AMD Instinct MI325、MI300 和 MI250）上，可以通过模拟方式运行 MXFP4/MXFP6 的矩阵乘法：权重会从 FP4/FP6 动态反量化为 half 精度，并采用融合 kernel。这样可用于在 vLLM 中评估 FP4/FP6 模型，或者用以获得约 2.5-4 倍的显存节省（相较于 float16 和 bfloat16）。

要生成 MXFP4 数据类型的离线量化模型，推荐直接使用 AMD Quark 的 [量化脚本](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)。示例：

```bash
python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --quant_scheme w_mxfp4_a_mxfp4 \
    --output_dir qwen_1.5-moe-a2.7b-mxfp4 \
    --skip_evaluation \
    --model_export hf_format \
    --group_size 32
```

目前的集成支持 [FP4、FP6_E3M2、FP6_E2M3 的任意组合](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py)，可用于权重或激活。部分目标硬件（如 AMD Instinct MI350/MI355）已支持混合精度 GEMM，例如激活用 FP6、权重用 FP4。