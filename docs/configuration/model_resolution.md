# 模型解析

vLLM 通过检查模型仓库中的 `config.json` 文件里的 `architectures` 字段，来加载兼容 HuggingFace 的模型，并寻找已在 vLLM 注册的对应实现。
不过，模型解析可能会因为以下原因失败：

- 模型仓库的 `config.json` 文件缺少 `architectures` 字段。
- 非官方仓库可能会使用 vLLM 未记录的其他名称来指代模型。
- 同一个架构名称被多个模型共用，导致无法确定应该加载哪个模型。

为了解决这些问题，可以通过向 `hf_overrides` 选项传递 `config.json` 的覆盖内容，显式指定模型架构。
例如：

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

在我们的[支持的模型列表](../models/supported_models.md)中，可以查看 vLLM 能识别的模型架构。