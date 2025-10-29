# 基础模型

本指南将带你一步步实现一个基础的 vLLM 模型。

## 1. 引入你的模型代码

首先，从源码仓库克隆你所需的 PyTorch 模型代码。
比如，vLLM 的 [OPT 模型](../../../vllm/model_executor/models/opt.py) 就是基于 HuggingFace 的 [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) 文件改写的。

!!! warning
    请仔细阅读并遵循原始代码的版权及许可协议要求！

## 2. 让你的代码兼容 vLLM

为了确保你的模型能在 vLLM 上正常运行，需要满足以下要求：

### 初始化代码

模型中的所有 vLLM 模块构造函数都必须包含一个 `prefix` 参数。这个 `prefix` 通常是模块在模型参数字典中的完整名称，对以下场景非常重要：

- 运行时支持：vLLM 的注意力算子会根据完整的层名注册到模型参数中。每个注意力算子必须有唯一的前缀，以避免名称冲突。
- 非均匀量化支持：量化的 checkpoint 可以选择只量化部分层，其余保持全精度。通过在初始化时传入 `prefix`，vLLM 可以根据配置判断当前层是否需要量化。

初始化代码示例：

??? code

    ```python
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.attention import Attention

    class MyAttention(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.attn = Attention(prefix=f"{prefix}.attn")

    class MyDecoderLayer(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.self_attn = MyAttention(prefix=f"{prefix}.self_attn")

    class MyModel(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.layers = nn.ModuleList(
                [MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") for i in range(vllm_config.model_config.hf_config.num_hidden_layers)]
            )

    class MyModelForCausalLM(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            self.model = MyModel(vllm_config, prefix=f"{prefix}.model")
    ```

### 计算代码

- 在 `MyModel` 模块内部添加一个 `get_input_embeddings` 方法，用于根据 `input_ids` 返回文本嵌入。这样可以为多模态模型提供一致的接口，相当于直接调用嵌入层。

```python
class MyModel(nn.Module):
        ...

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ... 
```

- 重写模型的 [forward][torch.nn.Module.forward] 方法，去掉训练相关的冗余代码。把输入参数改为把 `input_ids` 和 `positions` 视为一维的扁平张量（只有 batch 维度，没有 max-sequence 长度维度）。

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

!!! note
    目前，vLLM 支持基础的多头注意力机制及其旋转位置嵌入（rotary positional embedding）变体。
    如果你的模型使用其他类型的注意力机制，你需要在 vLLM 中自行实现对应的注意力层。

参考我们的 [Llama 实现](../../../vllm/model_executor/models/llama.py)。vLLM 已经支持了大量模型，建议找一个与你的模型结构类似的实现，参考并改造。更多示例可见 [vllm/model_executor/models](../../../vllm/model_executor/models)。

## 3.（可选）实现张量并行和量化支持

如果你的模型太大，单块 GPU 无法容纳，可以使用张量并行（tensor parallelism）。
具体做法是将模型中的线性层和嵌入层替换为张量并行版本。
嵌入层可以直接用 `VocabParallelEmbedding` 替换 [torch.nn.Embedding][]，输出的 LM head 可以用 `ParallelLMHead`。
线性层则可选用以下方式并行化：

- `ReplicatedLinear`：输入和权重在多块 GPU 上完全复制，不节省显存。
- `RowParallelLinear`：输入张量按隐藏维度分块，权重矩阵按行（输入维度）分块，矩阵乘法后通过 all-reduce 合并结果。常用于 FFN 第二层和注意力层的输出线性变换。
- `ColumnParallelLinear`：输入张量复制，权重矩阵按列（输出维度）分块，输出也按列分块。常用于 FFN 第一层和 Transformer 原始注意力层的 QKV 变换。
- `MergedColumnParallelLinear`：将多个 `ColumnParallelLinear` 操作合并，常用于带加权激活函数（如 SiLU）的 FFN 第一层。此类会处理多权重矩阵的分布式加载逻辑。
- `QKVParallelLinear`：用于多头和分组查询注意力机制的 query、key、value 投影。若 key/value 头数少于世界大小，会自动复制 key/value 头。此类负责权重的加载和复制。

注意以上所有线性层都需要 `linear_method` 参数，vLLM 会根据不同量化方案设置该参数以支持权重量化。

## 4. 实现权重加载逻辑

你需要在自己的 `*ForCausalLM` 类中实现 `load_weights` 方法。
该方法负责从 HuggingFace 的 checkpoint 文件加载权重，并分配到模型的各个层。特别是 `MergedColumnParallelLinear` 和 `QKVParallelLinear` 层，如果原始模型有独立的权重矩阵，需要分别加载不同部分。

## 5. 注册你的模型

如何在 vLLM 内注册模型可参考 [本页面](registration.md)。

## 常见问题解答

### 如何支持交错滑动窗口（interleaving sliding windows）模型？

对于支持交错滑动窗口的模型（如 `google/gemma-2-2b-it` 和 `mistralai/Ministral-8B-Instruct-2410`），调度器会将其视为全注意力模型，即不会丢弃 kv-cache 中的任何 token。这是为了保证前缀缓存机制正常运行。滑动窗口只作为 attention kernel 的参数出现。

支持此类模型时，需要注意以下细节：

- 确保模型的 `config.json` 文件包含 `layer_types`。
- 在模型代码中，为每一层解析正确的滑动窗口参数，并通过 `per_layer_sliding_window` 参数传递给注意力层。参考 [此代码行](https://github.com/vllm-project/vllm/blob/996357e4808ca5eab97d4c97c7d25b3073f46aab/vllm/model_executor/models/llama.py#L171)。

这两步完成后，交错滑动窗口机制即可在模型上生效。

### 如何支持使用 Mamba 的模型？

我们将 Mamba 支持分为三种情况：

1. 仅包含 Mamba 层（Mamba-1 或 Mamba-2），不包含注意力层的模型。
2. 同时包含 Mamba 层（Mamba-1 或 Mamba-2）和注意力层的混合模型。
3. 同时包含类似 Mamba 机制（如 Linear Attention、ShortConv）与注意力层的模型。

对于第 (1) 种情况，建议参考 [`MambaForCausalLM`](../../../vllm/model_executor/models/mamba.py)（Mamba-1）或 [`Mamba2ForCausalLM`](../../../vllm/model_executor/models/mamba2.py)（Mamba-2）的实现。
模型需继承协议 `IsAttentionFree`，并实现类方法 `get_mamba_state_dtype_from_config` 和 `get_mamba_state_shape_from_config`，用于根据配置计算状态形状和数据类型。
Mamba 层建议使用 [`MambaMixer`](../../../vllm/model_executor/layers/mamba/mamba_mixer.py)（Mamba-1）或 [`MambaMixer2`](../../../vllm/model_executor/layers/mamba/mamba_mixer2.py)（Mamba-2）。
请不要再使用（已弃用的）`MambaCacheManager`，也不要复制任何 V0 版本特有的代码路径，这些旧代码很快会被移除。
此外，模型还需加入 [vllm/model_executor/models/config.py](../../../vllm/model_executor/models/config.py) 的 `MODELS_CONFIG_MAP` 字典，以优化运行时默认参数。

对于第 (2) 种情况，可以参考 [`JambaForCausalLM`](../../../vllm/model_executor/models/jamba.py)（Mamba-1 + 注意力示例）或 [`BambaForCausalLM`](../../../vllm/model_executor/models/bamba.py)（Mamba-2 + 注意力示例）。
这些模型实现方式与第 (1) 种类似，但需继承协议 `IsHybrid`（不再用 `IsAttentionFree`），而且不必添加到 `MODELS_CONFIG_MAP`（运行时默认参数会自动推断）。

对于第 (3) 种情况，可参考 [`MiniMaxText01ForCausalLM`](../../../vllm/model_executor/models/minimax_text_01.py) 或 [`Lfm2ForCausalLM`](../../../vllm/model_executor/models/lfm2.py) 的实现，它们分别使用自定义的 "mamba-like" 层 `MiniMaxText01LinearAttention` 和 `ShortConv`。
实现方法同第 (2) 种情况。
"Mamba-like" 指的是那些内部状态会原地更新，而不是像注意力 KV cache 一样追加的层。
如果你要实现新的自定义 mamba-like 层，应继承 `MambaBase`，并实现 `get_state_dtype` 和 `get_state_shape` 方法（用于运行时计算数据类型和状态形状），以及 `mamba_type` 和 `get_attn_backend`。
还需实现一个 "attention meta-data" 类，负责管理所有层通用的元数据。参考 [`LinearAttentionMetadata`](../../../vllm/v1/attention/backends/linear_attn.py) 或 [`ShortConvAttentionMetadata`](../../../vllm/v1/attention/backends/short_conv_attn.py)。
如果你希望支持 torch compile 和 CUDA graphs，需要将对 mamba-like 层的调用包裹在自定义算子内并注册。可参考 [vllm/model_executor/models/minimax_text_01.py](../../../vllm/model_executor/models/minimax_text_01.py) 或 [vllm/model_executor/layers/mamba/short_conv.py](../../../vllm/model_executor/layers/mamba/short_conv.py) 中 `direct_register_custom_op` 的用法。
最后，新的自定义算子应加入 [vllm/config/compilation.py](../../../vllm/config/compilation.py) 的 `_attention_ops` 列表，以确保分段 CUDA graph 能正常工作。
