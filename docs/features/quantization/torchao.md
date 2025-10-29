# TorchAO

TorchAO 是一个用于 PyTorch 的架构优化库，提供高性能的数据类型（dtype）、优化技术和内核（kernel），可用于推理和训练，支持与 PyTorch 原生特性（如 torch.compile、FSDP 等）灵活组合。部分性能基准可以在 [这里](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks) 查看。

我们推荐使用如下命令安装最新的 torchao 每日构建版本：

```bash
# 安装最新的 TorchAO 每日构建版
# 请根据你的系统选择合适的 CUDA 版本（如 cu126、cu128 等）
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

## HuggingFace 模型量化

你可以使用 torchao 对自己的 huggingface 模型进行量化，比如 [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) 和 [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao)，也可以将模型检查点（checkpoint）上传到 huggingface hub，参考如下代码示例：[示例链接](https://huggingface.co/jerryzh168/llama3-8b-int8wo)：

??? code

    ```Python
    import torch
    from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import Int8WeightOnlyConfig

    model_name = "meta-llama/Meta-Llama-3-8B"
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    hub_repo = # 你的 hub 仓库 ID
    tokenizer.push_to_hub(hub_repo)
    quantized_model.push_to_hub(hub_repo, safe_serialization=False)
    ```

此外，你还可以使用 [TorchAO Quantization space](https://huggingface.co/spaces/medmekk/TorchAO_Quantization) 的简易界面来量化模型。