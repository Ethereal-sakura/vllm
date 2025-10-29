# 支持的模型

vLLM 支持用于多种任务的 [生成式模型](./generative_models.md) 和 [池化模型](./pooling_models.md)。

针对每一种任务，我们都会列出 vLLM 已经实现的模型架构，并为每个架构附上一些主流的模型示例。

## 模型实现说明

### vLLM 原生支持

如果某个模型得到了 vLLM 的原生支持，其实现可以在 [vllm/model_executor/models](../../vllm/model_executor/models) 目录中找到。

这些模型被收录在 [支持的文本模型](#list-of-text-only-language-models) 和 [支持的多模态模型](#list-of-multimodal-language-models) 两个列表中。

### Transformers 支持

vLLM 还支持 Transformers 库中已有的模型实现。你可以预期，vLLM 中调用 Transformers 实现的模型，其性能与 vLLM 原生实现的模型相差不会超过 5%。我们把这种能力称为 "Transformers 后端"。

目前，Transformers 后端支持如下内容：

- 模态类型：嵌入模型、语言模型、视觉-语言模型*
- 架构类型：仅编码器、仅解码器、专家混合结构
- 注意力类型：全注意力、滑动注意力等

_*目前，视觉-语言模型仅支持图片输入，视频输入的支持将在未来版本中上线。_

如果 Transformers 模型实现遵循了[自定义模型编写指南](#writing-custom-models)的所有流程，那么在 Transformers 后端下使用时，可以兼容以下 vLLM 功能：

- 支持 [功能兼容矩阵](../features/README.md#feature-x-feature) 中的所有特性
- 支持如下 vLLM 并行方案的任意组合：
    - 数据并行
    - 张量并行
    - 专家并行
    - 流水线并行

判断当前模型后端是否为 Transformers 非常简单：

```python
from vllm import LLM
llm = LLM(model=...)  # 填入你的模型名称或路径
llm.apply_model(lambda model: print(type(model)))
```

如果输出类型以 `Transformers...` 开头，则说明该模型使用了 Transformers 实现！

如果一个模型既有 vLLM 实现，你又想用 Transformers 后端实现，可以在 [离线推理](../serving/offline_inference.md) 时加上 `model_impl="transformers"`，或在 [在线服务](../serving/openai_compatible_server.md) 时加上 `--model-impl transformers`。

!!! note
    对于视觉-语言模型，如果加载模型时设置了 `dtype="auto"`，vLLM 会根据 config 的 `dtype` 加载整个模型。而 Transformers 原生实现会按各自骨干网络的 `dtype` 属性来加载，可能导致性能略有差异。

#### 自定义模型

如果一个模型既没有被 vLLM 原生支持，也不在 Transformers 支持列表内，你依然可以在 vLLM 中使用它！

要让一个模型能被 vLLM 的 Transformers 后端兼容，需要满足：

- 是 Transformers 兼容的自定义模型（参考 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）：
    - 模型目录结构需规范（如包含 `config.json` 等）
    - `config.json` 中需包含 `auto_map.AutoModel`
- 满足 vLLM Transformers 后端定制要求（见 [自定义模型编写](#writing-custom-models)）：
    - 定制应在基础模型实现（如在 `MyModel`，而非 `MyModelForCausalLM`）

如果模型在 Hugging Face Model Hub 上，只需在 [离线推理](../serving/offline_inference.md) 时加 `trust_remote_code=True`，或在 [openai-compatible-server](../serving/openai_compatible_server.md) 时加 `--trust-remote-code`。

如果模型在本地目录，只需将目录路径作为 `model=<MODEL_DIR>` 传入 [离线推理](../serving/offline_inference.md)，或 `vllm serve <MODEL_DIR>` 用于 [openai-compatible-server](../serving/openai_compatible_server.md)。

这意味着，借助 vLLM 的 Transformers 后端，你可以在官方支持之前抢先体验新模型！

#### 自定义模型编写指南

本节介绍如何将 Transformers 兼容的自定义模型适配为 vLLM Transformers 后端兼容模型。（假定你已经有了一个 Transformers 兼容的自定义模型，参考 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）

让模型兼容 Transformers 后端，需满足：

1. 所有模块（从 `MyModel` 到 `MyAttention`）都能传递 `kwargs`
    1. 如果是仅编码器模型，还需在 `MyAttention` 中加上 `is_causal = False`
2. `MyAttention` 必须通过 `ALL_ATTENTION_FUNCTIONS` 调用注意力机制
3. `MyModel` 必须有 `_supports_attention_backend = True`

<details class="code">
<summary>modeling_my_model.py</summary>

```python

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):
    is_causal = False  # 仅对编码器模型加此项

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

</details>

模型加载流程大致如下：

1. 读取 config
2. 从 config 的 `auto_map` 加载 `MyModel` 类，并检查模型是否 `is_backend_compatible()`
3. `MyModel` 会加载到 [vllm/model_executor/models/transformers](../../vllm/model_executor/models/transformers) 目录下的某个 Transformers 后端类，并设置 `self.config._attn_implementation = "vllm"`，由此调用 vLLM 的注意力层

就是这么简单！

如果你的模型要支持 vLLM 的张量并行/流水线并行，可在 config 类中添加 `base_model_tp_plan` 和/或 `base_model_pp_plan`：

<details class="code">
<summary>configuration_my_model.py</summary>

```python

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

</details>

- `base_model_tp_plan` 是一个 `dict`，将完整层名模式映射到张量并行方式（目前支持 `"colwise"` 和 `"rowwise"`）
- `base_model_pp_plan` 是一个 `dict`，将直接子层名映射为输入/输出变量名的 `tuple` 列表：
    - 只需对非所有流水线阶段都存在的层进行配置
    - vLLM 假定只有一个 `nn.ModuleList`，此列表会在流水线各阶段分布
    - `tuple` 的第一个列表为输入参数名，最后一个列表为输出变量名

## 模型加载

### Hugging Face Hub

vLLM 默认从 [Hugging Face (HF) Hub](https://huggingface.co/models) 加载模型。如需变更模型下载路径，可设置 `HF_HOME` 环境变量，详细用法见 [官方文档](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)。

判断模型是否原生支持，可以查看 HF 仓库中的 `config.json`，如果 `"architectures"` 字段包含下文列表中的架构，则说明原生支持。

模型 **不一定** 要原生支持才可在 vLLM 中使用。
[Transformers 后端](#transformers) 让你能直接运行 Transformers 实现（甚至是 Hugging Face Hub 上的远程代码）！

!!! tip
    你可以直接运行以下代码来验证模型在运行时是否真的支持：

    ```python
    from vllm import LLM

    # 仅针对生成式模型（runner=generate）
    llm = LLM(model=..., runner="generate")  # 填入你的模型名称或路径
    output = llm.generate("Hello, my name is")
    print(output)

    # 仅针对池化模型（runner=pooling）
    llm = LLM(model=..., runner="pooling")
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    如果 vLLM 能够成功返回文本（生成式模型）或隐藏状态（池化模型），说明你的模型是支持的。

否则，请参阅 [添加新模型](../contributing/model/README.md) 获取在 vLLM 中实现新模型的方法，或 [在 GitHub 提交 issue](https://github.com/vllm-project/vllm/issues/new/choose) 申请支持。

#### 下载模型

你也可以用 Hugging Face CLI [下载模型](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download) 或模型仓库中的指定文件：

```bash
# 下载整个模型
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

# 指定缓存目录
huggingface-cli download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# 只下载模型仓库中的某个文件
huggingface-cli download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### 查看已下载模型

用 Hugging Face CLI 管理本地缓存中的模型：

```bash
# 列出已经缓存的模型
huggingface-cli scan-cache

# 显示更详细的输出
huggingface-cli scan-cache -v

# 指定缓存目录
huggingface-cli scan-cache --dir ~/.cache/huggingface/hub
```

#### 删除缓存模型

用 Hugging Face CLI 交互式删除已经下载的模型缓存：

<details>
<summary>命令示例</summary>

```console
# delete-cache 命令需要额外依赖，请先运行 pip install huggingface_hub[cli]

# 启动交互式 TUI 界面选择要删除的模型
$ huggingface-cli delete-cache
? Select revisions to delete: 1 revisions selected counting for 438.9M.
  ○ None of the following (if selected, nothing will be deleted).
Model BAAI/bge-base-en-v1.5 (438.9M, used 1 week ago)
❯ ◉ a5beb1e3: main # modified 1 week ago

Model BAAI/bge-large-en-v1.5 (1.3G, used 1 week ago)
  ○ d4aa6901: main # modified 1 week ago

Model BAAI/bge-reranker-base (1.1G, used 4 weeks ago)
  ○ 2cfc18c9: main # modified 4 weeks ago

按 <space> 选择，<enter> 确认，<ctrl+c> 退出不做任何修改。

# 选择后还需确认
? Select revisions to delete: 1 revision(s) selected.
? 1 revisions selected counting for 438.9M. Confirm deletion ? Yes
Start deletion.
Done. Deleted 1 repo(s) and 0 revision(s) for a total of 438.9M.
```

</details>

#### 使用代理下载

下面是使用代理加载/下载 Hugging Face 模型的一些技巧：

- 为当前会话全局设置代理（或写进 profile 文件）：

```shell
export http_proxy=http://your.proxy.server:port
export https_proxy=http://your.proxy.server:port
```

- 只为当前命令设置代理：

```shell
https_proxy=http://your.proxy.server:port huggingface-cli download <model_name>

# 或直接用 vllm 命令
https_proxy=http://your.proxy.server:port vllm serve <model_name>
```

- 在 Python 解释器中设置代理：

```python
import os

os.environ["http_proxy"] = "http://your.proxy.server:port"
os.environ["https_proxy"] = "http://your.proxy.server:port"
```

### ModelScope 支持

如需从 [ModelScope](https://www.modelscope.cn) 加载模型而非 Hugging Face Hub，可设置环境变量：

```shell
export VLLM_USE_MODELSCOPE=True
```

加载时加上 `trust_remote_code=True`：

```python
from vllm import LLM

llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)

# 生成式模型（runner=generate）
output = llm.generate("Hello, my name is")
print(output)

# 池化模型（runner=pooling）
output = llm.encode("Hello, my name is")
print(output)
```

## 功能支持标记说明

- ✅︎ 表示该模型已支持此功能
- 🚧 表示此功能计划支持，但尚未实现
- ⚠️ 表示该功能可用，但存在已知问题或限制

## 文本语言模型支持列表

### 生成式模型

如何使用生成式模型详见 [此页面](generative_models.md)。

#### 文本生成

这些模型主要支持 [`LLM.generate`](./generative_models.md#llmgenerate) API。对于对话/指令模型，还支持 [`LLM.chat`](./generative_models.md#llmchat) API。

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| 架构 | 模型名称 | 典型 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `ApertusForCausalLM` | Apertus | `swiss-ai/Apertus-8B-2509`, `swiss-ai/Apertus-70B-Instruct-2509` 等 | ✅︎ | ✅︎ |
| `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B` 等 | ✅︎ | ✅︎ |
| `ArceeForCausalLM` | Arcee (AFM) | `arcee-ai/AFM-4.5B-Base` 等 | ✅︎ | ✅︎ |
| `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct` 等 | | ✅︎ |
| `BaiChuanForCausalLM` | 百川2、百川 | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B` 等 | ✅︎ | ✅︎ |
| ... | ... | ... | ... | ... |

部分模型仅通过 [Transformers 后端](#transformers) 支持。下表是官方通过 Transformers 后端支持的模型名单。日志中会显示正在使用 Transformers 后端，且不会提示降级。如果这些模型出现问题，请[提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)，我们会积极修复！

| 架构 | 模型名称 | 典型 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `SmolLM3ForCausalLM` | SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | ✅︎ | ✅︎ |

!!! note
    vLLM 的 ROCm 版本目前仅支持 Mistral 和 Mixtral，最大上下文长度为 4096。

### 池化模型

更多池化模型使用方法请见 [此页面](./pooling_models.md)。

!!! important
    部分模型架构同时支持生成和池化任务，
    建议明确指定 `--runner pooling`，确保以池化方式加载模型。

#### 向量嵌入

这些模型主要支持 [`LLM.embed`](./pooling_models.md#llmembed) API。

| 架构 | 模型名称 | 典型 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `BertModel`<sup>C</sup> | BERT 系列 | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs` 等 | | |
| ... | ... | ... | ... | ... |

<sup>C</sup> 通过 `--convert embed` 自动转为嵌入模型。([详细用法](./pooling_models.md#model-conversion))  
\* 功能支持与原始模型一致

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` 配置不规范，需手动指定均值池化 `--pool