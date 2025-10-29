# 量化 KV 缓存（Quantized KV Cache）

## FP8 KV 缓存

将 KV 缓存量化为 FP8，可以大幅降低内存占用。这意味着缓存中可以存储更多的 token，从而提升整体吞吐量。

### FP8 格式

[OCP（开放计算项目，Open Compute Project）](https://www.opencompute.org) 定义了两种常见的 8 位浮点数据格式：

- E5M2（5 位指数，2 位尾数）
- E4M3FN（4 位指数，3 位尾数，通常简称为 E4M3）

E4M3 格式相比 E5M2 精度更高。但由于其动态范围较小（±240.0），E4M3 通常需要为每个量化张量额外配备一个高精度（FP32）缩放因子。

### 当前限制

目前只支持每个张量（标量）缩放因子。后续会逐步支持更细粒度的缩放因子（比如每通道）。

### 性能影响

当前的 FP8 KV 缓存实现，主要通过为 KV 缓存分配大约两倍空间，从而提升吞吐量。这带来的优势包括：

- 单次请求可处理更长的上下文，或
- 可同时处理更多的并发批次

目前延迟尚未降低，因为还没有实现解量化和注意力操作的融合。未来版本将会支持硬件加速的量化注意力，从而带来更高的性能。现在一些新一代芯片（比如 AMD MI300、NVIDIA Hopper 及之后的产品）已经支持 FP8 与其他格式（fp32、fp16、bf16）之间的硬件级转换，但目前还未完全释放这些硬件的全部潜力。

研究显示，FP8 E4M3 量化对推理精度影响很小，因此非常适合用于提升吞吐量。

## 使用示例

下面是开启 FP8 量化的示例：

??? code

    ```python
    # 开启 calculate_kv_scales 参数，可自动计算 kv 缓存缩放因子

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        kv_cache_dtype="fp8",
        calculate_kv_scales=True,
    )
    prompt = "London is the capital of"
    out = llm.generate(prompt, sampling_params)[0].outputs[0].text
    print(out)
    ```

参数 `kv_cache_dtype` 用于指定 KV 缓存的存储数据类型：

- `"auto"`：使用模型默认的“未量化”数据类型
- `"fp8"` 或 `"fp8_e4m3"`：支持 CUDA 11.8+ 及 ROCm（AMD GPU）
- `"fp8_e5m2"`：支持 CUDA 11.8+

## 更高精度的校准缩放因子

如果希望在使用 FP8 KV 缓存时获得更优的模型表现，推荐使用针对实际推理数据校准过的缩放因子。[LLM Compressor](https://github.com/vllm-project/llm-compressor/) 是推荐的工具。

### 安装方法

首先安装所需依赖：

```bash
pip install llmcompressor
```

### 使用示例

以下是以 `meta-llama/Llama-3.1-8B-Instruct` 为例的完整流程（大部分模型都能用类似方式）：

??? code

    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.transformers import oneshot

    # 选择模型并加载
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 选择校准数据集
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # 设置校准参数
    NUM_CALIBRATION_SAMPLES = 512  # 建议从 512 条样本开始
    MAX_SEQUENCE_LENGTH = 2048

    # 加载并预处理数据集
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def process_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

    # 配置量化策略
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: 8
                    type: float
                    strategy: tensor
                    dynamic: false
                    symmetric: true
    """

    # 执行量化
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # 保存量化后的模型，例如：Llama-3.1-8B-Instruct-FP8-KV
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

上面的脚本会在当前目录下生成一个包含量化模型（如 `Llama-3.1-8B-Instruct-FP8-KV`）及校准缩放因子的文件夹。

使用模型时，请确保指定 `kv_cache_dtype="fp8"`，以启用 kv 缓存量化及相应缩放因子。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="Llama-3.1-8B-Instruct-FP8-KV", kv_cache_dtype="fp8")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```