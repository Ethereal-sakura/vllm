# INT4 W4A16

vLLM 支持将权重量化为 INT4（4 位整型），可以显著节省内存并加速推理。这种量化方法非常适用于低查询每秒（QPS）场景，可以有效减小模型体积，同时保持低延迟。

欢迎访问 HF 上的 [适用于 vLLM 的主流大模型 INT4 量化模型集合](https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c) 

!!! note
    INT4 计算仅在 NVIDIA GPU 计算能力 > 8.0（Ampere、Ada Lovelace、Hopper、Blackwell）上支持。

## 使用前准备

要在 vLLM 中使用 INT4 量化功能，需先安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```bash
pip install llmcompressor
```

此外，为了评估模型，还需要安装 `vllm` 和 `lm-evaluation-harness`：

```bash
pip install vllm git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api]
```

## 量化流程

量化过程主要分为四个步骤：

1. 加载模型
2. 准备校准数据
3. 执行量化
4. 在 vLLM 中评估精度

### 1. 加载模型

使用标准的 `transformers` AutoModel 类加载你的模型和分词器：

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

### 2. 准备校准数据

在进行 INT4 权重量化时，需要使用样本数据来估算权重更新和校准比例。建议选用与实际部署场景相似的数据进行校准。
对于通用指令微调模型，可以选择如 `ultrachat` 这类数据集：

??? code

    ```python
    from datasets import load_dataset

    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # 加载并预处理数据集
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ```

### 3. 执行量化

现在可以开始使用量化算法对模型进行处理：

??? code

    ```python
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    # 配置量化算法
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

    # 执行量化
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # 保存压缩后的模型，如 Meta-Llama-3-8B-Instruct-W4A16-G128
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

执行完以上流程后，将得到一个权重已量化为 4 位整型的 W4A16 模型。

### 4. 精度评估

量化后，可以在 vLLM 中加载和运行模型：

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
```

模型评估可通过 `lm_eval` 工具进行：

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量化模型对 `bos` token 的有无较为敏感。评估时请务必添加 `add_bos_token=True` 参数。

## 最佳实践

- 校准数据建议从 512 条样本开始，若精度不理想可适当增加
- 选择样本多样性高的校准数据，避免模型只适用于某一特定场景
- 推荐初始序列长度为 2048
- 使用模型训练时相同的对话模板或指令模板
- 如有微调过的模型，可直接选用训练集中的部分样本做校准
- 针对量化算法调优关键超参数：
    - `dampening_frac` 控制 GPTQ 算法的影响力。数值较低可以提升精度，但可能导致数值不稳定，算法运行失败。
    - `actorder` 控制激活顺序。对权重量化时，通道的量化顺序很重要。设置 `actorder="weight"` 可以提升精度且不会增加延迟。

下面是一个可供自定义调优的量化 recipe 示例：

??? code

    ```python
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    ) 
    recipe = GPTQModifier(
        targets="Linear",
        config_groups={
            "config_group": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    symmetric=True,
                    dynamic=False,
                    actorder="weight",
                ),
            ),
        },
        ignore=["lm_head"],
        update_size=NUM_CALIBRATION_SAMPLES,
        dampening_frac=0.01,
    )
    ```

## 常见问题与支持

如果遇到问题或有功能需求，欢迎在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 仓库提交 issue。完整 INT4 量化示例代码可在 [此处](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_example.py) 获取。