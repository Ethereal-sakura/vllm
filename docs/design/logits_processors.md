# Logits Processors（Logits 处理器）

!!! important
    部分 logits 处理器的设计还在调整中，相关 API 近期可能会有变动。我们也希望可以尽快稳定这部分的 API

本文档介绍了 vLLM 引擎如何与 logits 处理器交互，以及 vLLM 支持的 logits 处理器编程模型。

## Logits 处理器基础

Logits 处理器用于调整下一个 token 的概率分布，通常是为了让模型产生更符合预期的行为。

在 vLLM 中，logits 处理器以 batch 为单位进行操作。在每一次引擎推理步骤中，logits 处理器会接收一个形状为 `(num_requests) x (vocab_size)` 的原始 logits 张量。对于启用了 logits 处理器的请求，处理器会对 logits 张量中对应的行进行转换，其他行则保持不变。转换后的 logits 张量会传递给 softmax 进行后续处理。

## vLLM 引擎中的 Logits 处理器

vLLM 引擎的持久化 batch 数据结构会维护已加载的 logits 处理器列表。

为了能在整个 batch 上操作，每个 logits 处理器可以维护关于 batch 内请求的元数据（比如每个请求专属的参数配置）。因此，logits 处理器是有状态的。

在每个引擎步骤，vLLM 引擎会执行以下操作：(1) 更新每个 logits 处理器的内部状态；(2) 对模型输出的 logits 应用 logits 处理器。

### 更新 Logits 处理器内部状态

在每个引擎步骤开始时，持久化 batch 可能会根据调度器的结果添加、丢弃或重新排序请求。batch 完成重组后，vLLM 引擎会调用每个 logits 处理器的 `update_state()` 方法。这是为了确保 logits 处理器的内部状态可以和新的 batch 状态同步。

下面的伪代码展示了 vLLM 如何通知 logits 处理器 batch 状态的变化：

??? code "Model Runner Updates Logits Processor States"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            self._update_states(scheduler_output)

            ...

        def _update_states(...):

            ...

            # ...update persistent batch to reflect new/finished requests & reordering
            # of requests within batch...

            ...

            self.input_batch.refresh_metadata()


    # gpu_input_batch.py

    class InputBatch:

        ...

        def refresh_metadata(self):

            ...

            # Update each logits processor's state to reflect persistent batch state
            batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
            for logit_proc in self.logitsprocs.all:
                logit_proc.update_state(batch_update)

            ...


    # vllm/v1/sample/logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # Batch state-change data structure which is passed to logits processors'
        # update_state() methods

        batch_size: int

        removed: Sequence[RemovedRequest]
        added: Sequence[AddedRequest]
        moved: Sequence[MovedRequest]
    
    ```

### 对模型输出 logits 应用 Logits 处理器

持久化 batch 状态更新后，vLLM 的模型 runner 会进行模型推理得到 logits。然后，模型 runner 会调用采样器，对 logits 进行采样。采样器的一部分工作是调用 logits 处理器的 `apply()` 方法，对模型输出的 logits 进行转换（`apply()` 方法可以就地或非就地修改 logits，通常就地修改更节省内存）。相关流程见下方伪代码。

注意，采样器会通过 `SamplingMetadata.logitsprocs` 访问 logits 处理器。当 vLLM 引擎构造 `SamplingMetadata` 时（下方代码未展示），logits 处理器列表会从持久化 batch 数据结构传递到 `SamplingMetadata`。

??? code "Apply logits processors to model output logits"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            # (discussed in previous section)
            self._update_states(scheduler_output)

            ...

            # ...run model inference to obtain logits...

            ...

            # Invoke sampler, which applies logits processors
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=sampling_metadata)

            ...


    # sampler.py

    class Sampler(nn.Module):

        ...

        def forward(self, logits, sampling_metadata):

            ...

            # Apply non-argmax-invariant logits processors to model output logits
            for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
                logits = processor.apply(logits)

            sampled = self.sample(logits, sampling_metadata)

            ...

            # ...return sampler output data structure...


        def sample(self, logits, sampling_metadta)

            ...

            # ...exit early if all requests are greedy-sampling...

            ...

            # Apply argmax-invariant logits processors
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # ...perform sampling and return sampling result...
    ``` 

在采样时，采样器会检查 batch 中所有请求是否都采用贪婪采样（greedy sampling）。如果是，采样器会跳过所有“argmax 不变（argmax-invariant）”的 logits 处理器，以节省计算资源。这里的“argmax”指的是 logits 张量每一行中值最大的 token ID（即模型对该请求打分最高的 token）。

* **argmax 不变（argmax-invariant）的 logits 处理器**：比如 Min-P 等，它们不会改变最大值对应的 token id。例如，只屏蔽概率最低的 token，不会影响最大 logits 的 token。贪婪采样总是选择最大 logits 的 token，因此这种处理器在贪婪采样下可以跳过。

* **非 argmax 不变（non-argmax-invariant）的 logits 处理器**：可能会影响 argmax。例如，某些处理器会在指定步数后屏蔽除 EOS 之外的所有 token，以强制结束生成，这种操作可能会屏蔽最大 logits 的 token，从而改变 argmax。这类处理器即使在贪婪采样下也不能省略。

vLLM 对 logits 处理器的抽象要求在整个 batch 上应用，因此实际上只有当整个 batch 都是贪婪采样时，才可以跳过 argmax-invariant 的 logits 处理器。

## Logits 处理器编程模型

前面的章节已经提到 vLLM 的 logits 处理器需要实现哪些接口。本节会详细介绍如何实现与 vLLM 引擎兼容的 logits 处理器，包括 `LogitsProcessor` 基类、接口方法，以及用于表示 batch 状态变化的 `BatchUpdate` 数据结构，相关代码如下：

??? code "`LogitsProcessor` base class and `BatchUpdate` data structure"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING

    import torch

    from vllm import SamplingParams

    if TYPE_CHECKING:
        from vllm.config import VllmConfig


    class MoveDirectionality(Enum):
        # One-way i1->i2 req move within batch
        UNIDIRECTIONAL = auto()
        # Two-way i1<->i2 req swap within batch
        SWAP = auto()


    # (index, params, prompt_tok_ids, output_tok_ids) tuples for new
    # requests added to the batch.
    AddedRequest = tuple[int, SamplingParams, list[int], list[int]]

    # (index 1, index 2, directionality) tuples representing
    # one-way moves or two-way swaps of requests in batch
    MovedRequest = tuple[int, int, MoveDirectionality]

    # Batch indices of any removed requests.
    RemovedRequest = int


    @dataclass(frozen=True)
    class BatchUpdate:
        """Persistent batch state change info for logitsprocs"""
        batch_size: int  # Current num reqs in batch

        # Metadata for requests added to, removed from, and moved
        # within the persistent batch.
        #
        # Key assumption: the `output_tok_ids` list (which is an element of each
        # tuple in `added`) is a reference to the request's running output tokens
        # list; via this reference, the logits processors always see the latest
        # list of generated output tokens
        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]


    class LogitsProcessor(ABC):

        @abstractmethod
        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool) -> None:
            raise NotImplementedError

        @abstractmethod
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def is_argmax_invariant(self) -> bool:
            """True if logits processor has no impact on the
            argmax computation in greedy sampling.
            NOTE: may or may not have the same value for all
            instances of a given LogitsProcessor subclass,
            depending on subclass implementation.
            """
            raise NotImplementedError

        @abstractmethod
        def update_state(
            self,
            batch_update: "BatchUpdate" | None,
        ) -> None:
            """Called when there are new output tokens, prior
            to each forward pass.

            Args:
                batch_update is non-None iff there have been
                changes to the batch makeup.
            """
            raise NotImplementedError
            
    ```

一个 vLLM logits 处理器必须继承自 `LogitsProcessor` 并至少实现以下方法：

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`：引擎配置对象
    * `device`：硬件加速设备信息
    * `is_pin_memory`：是否支持 pin memory，用于优化 logits 处理器实现

* `apply(self, logits: torch.Tensor) -> torch.Tensor`：
    * 输入一个形状为 `(num_requests) x (vocab_size)` 的 logits 张量（`logits`）
    * 在 batch 维度上应用 logits 处理器的变换
    * 返回变换后的 logits 张量
    * 可以选择就地修改（in-place）或非就地修改（out-of-place），就地通常更节省内存

* `is_argmax_invariant(self) -> bool`：
    * 如果 logits 处理器不会改变最大 logits 对应的 token id，返回 `True`；如果可能改变 argmax，返回 `False`
    * 该方法只会在启动时调用一次；如果返回 `True`，vLLM 会在全部请求为贪婪采样时跳过此 logits 处理器

* `update_state(self, batch_update: "BatchUpdate" | None) -> None`：
    * 输入一个表示 batch 状态变化的 `BatchUpdate` 结构体，在每个引擎步骤开始时调用
    * 利用 `BatchUpdate` 成员更新 logits 处理器的内部状态
    * **注意**：如果 batch 没有变化，`batch_update` 可能为 `None`。此时 LogitsProcessor 也可以根据自身 retained 的 `output_token_ids` 进行状态更新

### `BatchUpdate` 数据结构

`BatchUpdate` 把持久化 batch 建模为请求列表，支持如下三种状态变更操作（以下顺序即为 `update_state()` 中处理顺序）：

* **Remove（移除）**：删除指定下标 `i` 的请求，无替换

    * 在 `Batchupdate.removed` 中用一个 `int` 表示要移除的位置

    * 删除后的效果：

        ``` text
        Batch: [A,B,C]
        Remove @ i:  1

        =>

        New Batch: [A,x,C] # 移除 B，留下空位
        ```

* **Add（新增）**：在下标 `i` 处新增（或替换）请求。如果替换了已有请求，原有状态应被丢弃。

    * 在 `Batchupdate.added` 中用一个四元组表示：

        ``` text
        (index, new request SamplingParams, prompt token ids, output token ids)
        ```

    * `prompt token ids` 和 `output token ids` 分别是请求的 prompt 和输出 token id 列表的引用。注意，output token ids 会随着步数增长，logits 处理器可实时看到最新生成的 token。**对于依赖已生成 token 的 LogitsProcessor，这一点很关键。**

    * 具体如何处理 added 元组中的字段，由 logits 处理器子类实现。例如，不需要用到 prompt 或 output token ids 的处理器，可以只用 `index` 和 `SamplingParams`。

    * 如果 index `i` 已有请求，则替换：

        ``` text
        Batch: [A,B,C]
        新请求 D 加入 @ 1

        =>

        New Batch: [A,D,C] # D 替换 B
        ```

    * 如果 index `i` 超出当前 batch 长度，则为扩展：

        ``` text
        Batch: [A,B,C]
        新请求 D 加入 @ 3

        =>

        New Batch: [A,B,C,D] # 扩展 batch，加入 D
        ```

* **Move（移动）**：将下标 `s` 的请求移动到 `d`，或将 `s` 和 `d` 交换

    * 在 `Batchupdate.moved` 中用一个三元组表示：

        ``` text
        (s, d, UNIDIRECTIONAL or SWAP)
        ```

    * 如果是 `UNIDIRECTIONAL` 移动：

        * 把 `s` 的请求移到 `d`，`s` 位置变为空

            ``` text
            Batch: [A,x,C,D]
            单向移动 3 -> 1

            =>

            New Batch: [A,D,C,x] # D 移到 1，3 变空
            ```

        * 如果 `d` 处已有请求，则被替换丢弃

            ``` text
            Batch: [A,B,C,D]
            单向移动 3 -> 1

            =>

            New Batch: [A,D,C,x] # D 移到 1，B 被丢弃，3 变空
            ```

    * 如果是 `SWAP`，则交换 `s` 和 `d` 的请求

        ``` text
        Batch: [A,B,C,D]
        交换 3 <-> 1

        =>

        New Batch: [A,D,C,B] # 交换 B 和 D
        ```

此外，`BatchUpdate` 还包含一个 `batch_size` 字段，表示当前 batch 的大小。

### vLLM 引擎构建 `BatchUpdate` 的流程

logits 处理器的 `update_state()` 方法需要假定模型 runner 按如下方式更新 batch 状态（以 `BatchUpdate` 抽象表示）：

1. 找到本轮已完成的请求下标

2. 找到本轮新加入的请求

3. 用 Add 操作按序替换已完成请求

4. 根据新请求和完成请求的数量：

    1. 新旧请求数相等，继续下一步

    2. **新请求多于完成请求：** 用 Add 操作将多余的新请求添加到 batch 尾部，并分配连续下标

    3. **新请求少于完成请求：**
        * 对未被新请求替换的完成请求执行 Remove 操作，这些下标一定大于之前被替换的最大下标。此时 batch 可能出现空位
        * **“压缩” batch 使其连续：** 从最小的空位起，依次用单向 Move 把当前最大的非空位请求移到空位，直到 batch 连续
        * **调整 batch 大小：** 压缩后，所有空位集中到数组末尾，更新 `BatchUpdate.batch_size` 只包含非空位数量

5. 为优化效率，可能还会用 Swap Move 操作对 batch 进行重排

注意事项：

* logits 处理器的 `update_state()` 必须严格按 removes、adds、moves 的顺序处理 batch 更新操作

* Add 操作的 index 指的是 Add 发生时的下标，即执行 Move 之前的下标
    * 例如：如果一个请求先被 Add 到 5，然后和 3 交换，Add 的 index 还是 5
    * 换句话说，Move 操作一定是在 Add 和 Remove 之后

* Move 操作按 `BatchUpdate.moved` 的顺序依次执行

* 如果没有新增/完成请求，也没有 batch 重排，那么 logits 处理器收到的 batch update 会是 `None`

#### 示例：新请求少于完成请求的 batch 更新

下面例子演示了本轮只加入 1 个新请求、淘汰 2 个完成请求，并且 attention 后端还做了一次交换以优化 batch 顺序。

``` text
Batch 状态（step 开始）: [A,B,C,D]
Batch size: 4

新请求: E

完成请求: A, C

处理步骤（用 BatchUpdate 描述）:

1. 在 index 0 处 Add E

[E,B,C,D] # A 被替换
Batch size: 4

2. 在 index 2 处 Remove

[E,B,x,D] # C 被移除，2 为空
Batch size: 4

3. 用单向 Move 3 -> 2 压缩 batch，并调整 batch 大小

[E,B