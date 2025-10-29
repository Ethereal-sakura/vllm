# CUDA Graphs

本文将为你介绍 vLLM v1 中全新的 CUDA Graphs 模式，这些模式是在之前 [torch.compile 集成](torch_compile.md)的基础上新增的。简要来说，我们做了如下改进：

1. 新增了灵活的 `cudagraph_mode` 配置参数
2. 让 CUDA Graphs 的完整支持与编译过程解耦
3. 引入了 CUDA Graphs 分发器，作为统一的控制中心，可根据每个 batch 自动选择和调度合适的运行模式及 CUDA Graphs

本文将围绕以下内容展开：

* [设计动机](#motivation)
* [CUDA Graphs 模式](#cudagraphmodes)
* [详细设计](#detailed-design)
* [不同模式下的使用示例](#usage-guide)

!!! note
    本文中，纯解码（`max_query_len=1`）或 speculative 解码（`max_query_len =1+num_spec_tokens`）的 batch 被称为**同构解码（uniform decode）**，而其他类型的 batch（如 prefill 或 prefill+decode 混合）则归为**异构（non-uniform）**。

!!! note
    以下内容主要参考于 <https://github.com/vllm-project/vllm/pull/20059> 的最后一次提交。

## 设计动机

最初的分段（piecewise）编译是为了实现分段的 CUDA Graphs 捕获（cudagraph capture），从而排除掉不支持 CUDA Graphs 的操作（主要是 attention），这样可以在兼容所有 attention 后端的前提下获得一定的加速。后来我们又支持了“完整 CUDA Graphs”，即不再分段编译，这样在 attention 支持 CUDA Graphs 时可以进一步降低延迟。但这种编译与 cudagraph 捕获的强耦合，导致了灵活性受限，只能二选一。许多 attention 后端也未准备好统一支持“完整” CUDA Graphs（目前只有 FlashAttention 3 支持），或者只对纯解码 batch 支持 CUDA Graphs（比如 Flashinfer、FlashMLA、Mamba 等）。这带来了性能/兼容性的权衡变得混乱、支持策略不统一、代码结构也越来越复杂。

因此，我们希望实现更细粒度的 CUDA Graphs 方案，核心目标包括：

* 明确区分 prefill/mixed 与解码（decode）batch，并分别捕获 CUDA Graphs
* 尽量让 CUDA Graphs 捕获逻辑与编译解耦，这意味着：
    * 分段和完整 cudagraph 均可共用同一份编译好的图
    * 也可以只做完整 cudagraph 捕获而无需编译
* 运行时可根据 batch 组成在完整和分段 cudagraph 间动态切换
* 通过集中式的 cudagraph 控制，简化代码，方便扩展

这些能力让 cudagraph 捕获和编译可以灵活应对不同的启动/性能权衡和特性支持。

## `CudagraphModes`

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode] 是你在 `CompilationConfig.cudagraph_mode` 中唯一需要调整的参数：

* `NONE` — 关闭 CUDA Graphs，适合调试用。
* `PIECEWISE` — 单一分段模式（过去的默认），最灵活：不支持 CUDA Graphs 的操作（如 attention）走 eager，其他的则进入 CUDA Graphs。需要分段编译支持。
* `FULL` — 单一完整模式，仅对异构 batch 捕获完整 CUDA Graphs，同构解码 batch 复用相同 batch_size 下的异构 batch 的 CUDA Graph（因为兼容）；适合小模型或 prompt 较短的场景。
* `FULL_DECODE_ONLY` — 只对同构解码 batch 捕获完整 CUDA Graph，prefill/mixed 等则不用 cudagraph；适合解码为主、prefill 不重要的 P/D 场景，这样可以节省 `PIECEWISE` 模式下的显存。
* `FULL_AND_PIECEWISE` —（默认）同构解码 batch 用完整 CUDA Graph，其余用分段 CUDA Graph；通常性能最佳，尤其适合低延迟的小模型或 MoE，但显存占用最大，捕获时间也最长。

默认逻辑：如果你在 v1 中启用了分段编译，默认采用 `FULL_AND_PIECEWISE` 以获得更好性能（对于 pooling 模型还是 `PIECEWISE`）。如果当前不支持分段编译，则默认 `NONE`。

其中，`NONE`、`PIECEWISE`、`FULL` 都是单一模式，分别对应过去的 eager、分段 CUDA Graphs、完整 CUDA Graphs；而 `FULL_DECODE_ONLY` 与 `FULL_AND_PIECEWISE` 是新增的双模式，需要运行时根据 batch 动态切换。

!!! note
    这里 `NONE`、`PIECEWISE`、`FULL` 都会被视为分发器可选的 runtime mode。如果选择双模式，分发器会根据 batch 具体情况在其子模式间（以及必要时的 `NONE`）动态切换。

目前 cascade attention 虽然不支持 cudagraph，但它已兼容所有 cudagraph 配置。如果遇到 cascade attention，分发器会优先调度到 `PIECEWISE`，若不可用则退回 `NONE`。

!!! note
    并非所有 attention 后端都兼容所有 CUDA Graph 模式。我们会自动将当前模式“降级”到最近支持的。例如，如果后端只支持纯解码/同构 batch 的 CUDA Graphs，且已启用分段编译，则 `FULL` 会转为 `FULL_AND_PIECEWISE`，否则转为 `FULL_DECODE_ONLY`。

## 详细设计

### 总览

全新的 CUDA Graphs 逻辑是建立在分段编译之上的，支持运行时双模式切换。整体包括以下核心组件：

* [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper]：负责包装可调用对象，管理 CUDA Graphs 的捕获与复现
* [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]：统一的调度中心，负责各类 CUDA Graphs 的分发与管理
* [CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]：枚举，描述支持的所有模式（见上文）
* [BatchDescriptor][vllm.forward_context.BatchDescriptor]：用于唯一标识 runtime batch 的核心结构，作为分发 key

下图展示了 CUDA Graphs 与 inductor 编译在新旧设计中的对比。可以看到，旧版逻辑中 CUDA Graphs 与编译强耦合在 vllm `PiecewiseBackend` 内，并且仅以 batch_size 为 key 被动分发。而新版中，CUDA Graphs 逻辑拆分为独立的 `CUDAGraphWrapper`，专门负责完整和分段 CUDA Graphs，分发则通过显式的 runtime mode 和 `BatchDescriptor` 作为 key，由 `CudagraphDispatcher` 完成。

**旧设计：**

![previous_design](../assets/design/cuda_graphs/previous_design.png)

**新设计：**

![new_design](../assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor`

[BatchDescriptor][vllm.forward_context.BatchDescriptor] 是 `ForwardContext` 内的一个结构，和 CUDA Graphs 的 runtime mode 一起，用于运行时分发的 key。其简化原型如下：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
```

其中 `num_tokens` 代表补齐后的 token 长度，`uniform_decode` 依赖于 batch 的 `max_query_len` 是否等于目标同构解码的 `max_query_len`，并且 `num_scheduled_tokens` 能否被该值整除。

这个结构的目标是用最少的字段唯一标识一个（补齐后的）batch，以对应一个 CUDA Graphs 实例。像 `uniform_query_len` 这类信息可以忽略，因为在当前 setup 下它在运行时是常量。例如，常见的纯解码就是 `1`，speculative 解码验证阶段则为 `1+num_spec_tokens`。

!!! note
    未来 `BatchDescriptor` 可能会更通用，比如加入 `uniform_query_len` 支持多种同构解码长度等（见 <https://github.com/vllm-project/vllm/pull/23679>），或适配输入并不以 token 长度为主的模型场景（如多模态输入等）。

### `CudagraphDispatcher`

[CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher] 负责维护两组有效分发 key：一组用于 `FULL` 模式，一组用于 `PIECEWISE`，并在模型执行前根据 batch 的描述分发到对应模式和 key。分发器会根据初始 key（即补齐输入的 batch_descriptor），返回选定的 runtime mode 和最终的 batch_descriptor，然后通过 forward context 通知 CUDAGraphWrapper 该决策。注意，`CudagraphDispatcher` 是唯一可信的 CUDA Graphs key 管理者，`CUDAGraphWrapper` 只需信任 forward context 的分发结果即可。这让 wrapper 代码更简单，所有逻辑集中于分发器。

分发 key 由分发器的 `initialize_cudagraph_keys` 方法初始化，该方法在所有 attention 后端初始化后由 gpu_model_runner 调用。未来这里可扩展支持更多 CUDA Graphs 组合。目前只是按 `cudagraph_mode` 的 `decode_mode`/`mixed_mode` 及 compilation config 的 `cudagraph_capture_sizes` 组合生成可用 key。

分发相关代码示例：

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# 执行阶段
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
     output = self.model(...)
```

在 `dispatch()` 方法内部，分发器会优先查找 `FULL`、`PIECEWISE`、`None` 这三类现有 key。若找不到匹配的 key，则默认返回 `NONE`，即走 eager 执行。具体实现见 [这里](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91)。

下面是模型执行时分发流程的简化示意图：
![executor_runtime](../assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper`

[CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper] 用于包装可执行对象，为其添加 CUDA Graph 捕获和复现能力。每个 wrapper 绑定一个具体的 `runtime_mode`（只能是 `PIECEWISE` 或 `FULL`），负责捕获/复用 CUDA Graph 或直接调用原对象。运行时每个 wrapper 会：

1. 检查 global forward context 中的 runtime_mode 和 batch_descriptor（分发 key）
2. 若 runtime_mode 为 `NONE` 或与本 wrapper 模式不匹配，则直接调用
3. 否则，如果 runtime_mode 与本 wrapper 匹配，则进行 CUDA Graph 捕获（若无缓存则新建并缓存）或复现（若有缓存则复用）

上述流程的前提是 wrapper 会直接信任 forward context（即分发器的决策），这样可极大简化和集中逻辑，降低 wrapper 与 dispatcher 状态不同步的风险，并且 wrapper 类可复用于 `FULL` 和 `PIECEWISE` 模式。具体实现见 [这里](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106)。

#### 嵌套 Wrapper 设计

让完整 CUDA Graphs 与分段 CUDA Graphs 能共存的核心机制就是嵌套 CUDA Graphs wrapper 设计——基于分段编译、只需一个分段 FX 图即可。在模型最外层包上 FULL 模式 wrapper 实现完整 CUDA Graphs，模型内部各分段 backend 用 `PIECEWISE` 模式 wrapper 包装。

下图清晰展示了工作流程：
![wrapper_flow](../assets/design/cuda_graphs/wrapper_flow.png)

因此，对于 `FULL` runtime mode，可以安全地捕获/复现完整 CUDA Graph（此时分段 wrapper 不会被激活）；对于 `PIECEWISE`，只激活分段 wrapper，两者互不干扰。`NONE` 模式下，二者都不激活，直接 eager 执行。

### 完整 CUDA Graph 捕获与预热

CUDA Graphs 的捕获发生在 runner 首次以非 `NONE` runtime mode 调用模型 forward（通过 `_dummy_run`）时。对于完整 CUDA Graph 捕获，会通过设置 attention metadata 精确区分 prefill/mixed batch 与同构解码 batch，确保底层 attention 后端调度所需的 kernel。区分的关键是 attn_metadata 中的 `max_query_len`（大多数 attention 后端适用），同构解码时设置为目标 `uniform_query_len`，否则就是 batch 的 `num_tokens`。

CUDA Graphs wrapper 不再负责预热逻辑，预热现在由 GPU model runner 直接控制，会用 `NONE` runtime mode 进行 eager 执行预热。完整 CUDA Graph 捕获时，预热的 dummy_run 调用也要确保 attention 被覆盖到。

## 各 Attention 后端的 CUDA Graphs 支持

为标明 attention 后端对 CUDA Graphs 的支持情况，我们引入了枚举类型 [AttentionCGSupport][vllm.v1.attention.backends.utils.AttentionCGSupport]，用于描述后端支持的能力，值按能力强弱排序：`ALWAYS` > `UNIFORM_BATCH` > `UNIFORM_SINGLE_TOKEN_DECODE` > `NEVER`。

```python
class AttentionCGSupport(enum.Enum):
    """ Attention 后端的 CUDA Graphs 支持常量
    不考虑 cascade attention，因为它当前永远不支持 CUDA Graphs."""

    ALWAYS = 3
    """始终支持 CUDA Graphs，包括混合 prefill-decode"""
    UNIFORM_BATCH = 2
    """仅支持所有 query length 相同的 batch，比如 spec-decode
        即“解码”是 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """仅支持 query_len==1 的纯解码 batch"""
    NEVER = 0
    """完全不支持 CUDA Graphs"""
```

如果模型包含多种 attention backend（如 mamba mixer），我们取所有后端中的最弱能力，作为最终模型的能力，然后按需降级 CUDA Graphs 模式。例如能力最低为 `UNIFORM_BATCH` 时，`FULL` 会降级为 `FULL_AND_PIECEWISE`，能力最低为 `NEVER`（如 -O3 编译）则降级为 `PIECEWISE`。完整的回退策略可参考 [这里][vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode]。

下表为截至目前支持完整 CUDA Graphs 的部分后端：

| Attention 后端 | cudagraph_support | 说明 |
|:---|:---|:---|
| FlashAttention v2 | `UNIFORM_BATCH` | 实际上是 `ALWAYS`，但为性能考虑降级为 `FULL_AND_PIECEWISE` |
| FlashAttention v3 | `ALWAYS` | 解码和 prefill/mixed 共用统一 kernel，`FULL` 模式表现优秀 |
| Triton Attention | `ALWAYS` | 推荐用 `FULL_AND_PIECEWISE`，因解码和 prefill/mixed 用不同 kernel |
| AITER FlashAttention | `UNIFORM_BATCH`| |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| FlashMLA | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention| `UNIFORM_SINGLE_TOKEN_DECODE` | |

未列出的后端均视为 `NEVER`。

## 使用指南

现在 CLI 直接用大写字符串设置 cudagraph_mode，如：`--compilation-config '{"cudagraph_mode": "..."}'`，其中 `...` 取值为 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`。注意所有 `PIECEWISE` 相关模式都要求分段编译，所有 `FULL` 相关模式都需要 attention 后端支持 CUDA Graphs。例如：

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python 示例

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
)
sampling_params = vllm.SamplingParams(
    temperature=0,  # 贪婪解码
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### 从旧参数迁移

原有的 `use_cudagraph` 和 `full_cuda_graph` 现已