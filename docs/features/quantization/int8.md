# INT8 W8A8

vLLM 支持将权重和激活（activation）量化为 INT8，以节省显存并加速推理。这种量化方法特别适合在保持良好性能的同时，大幅减小模型体积。

欢迎访问 HuggingFace 上的 [vLLM 可直接使用的主流大模型 INT8 量化模型集合](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415) 。

!!! note
    INT8 计算仅支持计算能力大于 7.5 的 NVIDIA GPU（Turing、Ampere、Ada Lovelace、Hopper 架构）。

!!! warning
    **Blackwell GPU 限制**：计算能力大于等于 100 的 GPU（如 RTX 6000 Blackwell）不支持 INT8。请使用 [FP8 量化](fp8.md) ，或在 Hopper/Ada/Ampere 架构上运行。

## 前置条件

如需在 vLLM 中使用 INT8 量化，需要先安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```bash
pip install llmcompressor
```

此外，建议安装 `vllm` 和 `lm-evaluation-harness` 以便进行模型评测：

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

通过标准的 `transformers` AutoModel 类加载你的模型和分词器：

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

当你对激活进行 INT8 量化时，需要一些样本数据用于估算激活的缩放比例。建议选择与实际部署场景相似的数据进行校准。

对于通用的指令微调模型，可以使用如 `ultrachat` 这样的数据集：

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

</details>

### 3. 执行量化

现在可以应用量化算法：

??? code

    ```python
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    # 配置量化算法
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # 执行量化
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # 保存压缩后的模型，例如：Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

经过上述流程后，你将得到一个权重和激活均量化为 8 位整数的 W8A8 模型。

### 4. 评估精度

量化后，可以在 vLLM 中加载并运行模型：

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

如需评测模型精度，可用 `lm_eval` 工具：

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量化模型对 `bos` token 是否存在较为敏感。评测时请务必添加 `add_bos_token=True` 参数。

## 最佳实践建议

- 校准数据建议从 512 条样本起步（如精度下降可适当增加）
- 推荐的序列长度为 2048
- 使用模型训练时采用的聊天模板或指令模板
- 如模型经过微调，建议用你自己的训练数据样本做校准

## 故障排查与支持

如有问题或功能建议，请前往 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) 的 GitHub 仓库提交 issue。