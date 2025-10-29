# 自定义 Logits Processor

!!! important
    部分 logits processor 的设计还在持续迭代中，相关 API 在近期可能会有变化。我们希望尽快稳定这部分 API

“自定义” logits processor 是 vLLM 用户自己编写的，可以在初始化时加载到 vLLM，无需修改或重新编译 vLLM 源代码。它与内置 logits processor 相对。

本文将介绍如何编写、加载并使用自定义 logits processor。

## Logits Processor 背景知识

logits processor 主要用于调整下一个 token 的概率分布，通常目的是让模型输出更接近预期的行为。

在 vLLM 中，logits processor 是按批次（batch）粒度工作的。在每一步引擎执行时，logits processor 会处理一个形状为 `(num_requests) x (vocab_size)` 的原始 logits 张量（即模型输出）。对于启用了 logits processor 的请求，logits processor 会对对应的 logits 行进行变换，未启用则保持不变。变换后的 logits 会交给 softmax 处理。

## 如何创建自定义 Logits Processor

自定义的 logits processor 必须继承自 `vllm.v1.sample.logits_processor.LogitsProcessor`，并至少实现以下方法：

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`：引擎配置
    * `device`：硬件加速设备信息
    * `is_pin_memory`：是否支持 pin memory，辅助 logits processor 实现

* `apply(self, logits: torch.Tensor) -> torch.Tensor`：
    * 输入一个 `(num_requests) x (vocab_size)` 的 logits 张量
    * 对 batch 内所有请求进行 logits 变换操作
    * 返回变换后的 logits 张量（形状同输入）
    * 你可以选择原地（in-place）或非原地（out-of-place）修改 logits，原地修改更节省内存

* `is_argmax_invariant(self) -> bool`：
    * 如果 logits processor 保证不会改变每个请求的最大 logits（即 argmax）的 token id，则返回 `True`；否则返回 `False`
    * `is_argmax_invariant()` 在启动时只会被调用一次；如果返回 `True`，当所有请求都采用贪婪采样时，vLLM 会跳过这个 logits processor

* `update_state(self, batch_update: Optional["BatchUpdate"]) -> None`：
    * 接收一个 `BatchUpdate` 结构体，表示当前引擎步开始时 batch 的持久状态变化
    * 使用 `BatchUpdate` 内的信息更新 logits processor 的内部状态
    * **注意：** `batch_update` 可能为 `None`，表示 batch 成员没有变化。在这种情况下，LogitsProcessor 仍可以基于之前保存的 `output_token_ids` 列表更新内部状态

### vLLM 引擎如何构建 `BatchUpdate` 结构体

!!! important
    部分 logits processor 的设计还在持续调整。我们预计未来实现 logits processor 时无需关注 batch 状态变化，届时本节内容将不再适用。

实现 logits processor 的 `update_state()` 时，需要理解模型执行时如何更新 batch 的持久状态（以 `BatchUpdate` 抽象说明）：

1. 找出本步已完成的请求的索引

2. 找出本步新加入的请求

3. 用 Add 操作将新请求尽量替换已完成的请求，按被替换请求的索引递增顺序进行

4. 根据新请求和已完成请求的数量关系：

    1. 若数量相等，直接进入下一步

    2. *新请求多于已完成请求*：用 Add 操作将剩余新请求添加到 batch 末尾，并为这些新请求分配连续索引，从 `current_max_batch_index + 1` 开始

    3. *新请求少于已完成请求*：

        * 用 Remove 操作将未被新请求替换的已完成请求移除。被移除请求的索引必然大于前面被替换的请求的最大索引，Remove 可能导致 batch 索引不连续

        * **“压缩” batch：** 从最小的空槽（由 Remove 产生）开始，用单向 Move 将当前 batch 中最大索引的非空槽填充到空槽，继续按升序空槽索引、降序非空槽索引顺序执行 Move 直到 batch 连续

        * **缩减 batch 大小：** 压缩后，所有空槽会被移到 batch 数组末尾，最后更新 `BatchUpdate.batch_size` 以反映实际非空槽数量

5. 为提升效率，可能对 batch 进行排序。具体排序操作（Swap Move）数量与 attention 后端实现及 batch 特性有关

注意事项：

* `update_state()` 必须按以下顺序处理 batch 操作：先 Remove，再 Add，最后 Move

* Add 操作的 index 表示 *Add 发生时* 的索引（即未执行任何 Move 前的索引）
    * 例如：如果请求在索引 5 处 Add，然后与索引 3 互换，`BatchUpdate.added` 中 Add 仍记录为索引 5
    * 换句话说，Move 操作总是在 Add/Remove 之后执行

* Move 操作顺序同 `BatchUpdate.moved` 中的顺序

* 如果没有新请求/已完成请求，也没有 batch 排序，则 logits processor 收到的 batch update 会是 `None`

### 为自定义 Logits Processor 传递自定义参数

与内置 logits processor 不同，自定义 logits processor 可能需要配置一些未硬编码在 `SamplingParams` 或 vLLM 服务器 REST API 中的参数。为此，可以参考 vLLM [自定义参数](./custom_arguments.md) 的机制，让用户传递自定义配置（当然，你也可以让你的 processor 只用 `SamplingParams` 里的已有字段）。

### 自定义 Logits Processor 示例

下面是一个简单示例，实现了一个自定义 logits processor：它接收一个 `(num_requests) \times (vocab_size)` 的 logits 张量，只保留每个请求指定的 `target_token`，其他 token logits 设为 `float(-inf)`。如果某个请求未指定 `target_token`，则该请求不启用此 logits processor。是否启用以及保留哪个 token，由 logits processor 检查 `SamplingParams.extra_args` 对应的 `target_token` 参数决定：

??? code "自定义 logits processor 示例"

    ``` python
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (BatchUpdate,
                                                LogitsProcessor,
                                                MoveDirectionality)

    class DummyLogitsProcessor(LogitsProcessor):
        """伪造的 logits processor，用于单元测试和示例"""

        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool):
            self.req_info: dict[int, int] = {}

        def is_argmax_invariant(self) -> bool:
            """不会影响贪婪采样"""
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            # 处理新增请求
            for index, params, _, _ in batch_update.added:
                assert params is not None
                if params.extra_args and (target_token :=
                                        params.extra_args.get("target_token")):
                    self.req_info[index] = target_token
                else: 
                    self.req_info.pop(index, None)

            if self.req_info:
                # 处理移除的请求
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # 处理请求的移动（单向 move 或 swap）
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # 修改前保存目标 token 的值
            cols = torch.tensor(
                list(self.req_info.values()), dtype=torch.long, device=logits.device
            )
            rows = torch.tensor(
                list(self.req_info.keys()), dtype=torch.long, device=logits.device
            )
            values_to_keep = logits[rows, cols].clone()

            # 除目标 token 外全部 mask
            logits[rows] = float('-inf')
            logits[rows, cols] = values_to_keep

            return logits
    ```

在本文后续示例中，都将以 `DummyLogitsProcessor` 作为自定义 logits processor 的代表。

`DummyLogitsProcessor.update_state()` 通过字典 `self.req_info` 维护 batch 内请求的“稀疏”表示：只有显式指定了 `target_token` 的请求才有字典项。`update_state()` 会根据 Add/Remove/Move 操作调整字典里的请求索引和 `target_token`（即 key 和 value）。

### 包装已有的“请求级” Logits Processor

虽然 vLLM 引擎要求 logits processor 按 batch 粒度工作，但有些用户希望沿用“请求级” logits processor（即只处理单个请求的实现方式）。这种情况在你用过 vLLM 0.x 版本时尤其常见，当时要求 logits processor 是一个 `Callable`，类型如下（见 [v0 文档](https://docs.vllm.ai/en/v0.10.1.1/api/vllm/logits_process.html)）：

``` python
RequestLogitsProcessor = Union[

    # (output token ids, logits tensor) -> logits tensor
    Callable[[list[int], Tensor], Tensor],

    # (prompt token ids, output token ids, logits tensor) -> logits tensor
    Callable[[list[int], list[int], Tensor], Tensor],
]
```

虽然 vLLM 新引擎不直接支持请求级 logits processor，但你可以通过封装（wrap）的方式，将现有的“请求级” Callable processor 适配成批处理（batch-level）logits processor。只要你的 Callable 满足上述类型注解即可；如果接口不同，需要你自己加一层封装。

只需继承 `AdapterLogitsProcessor`，即可包装你的请求级 logits processor。你需要重写 `AdapterLogitsProcessor.is_argmax_invariant(self)`，反映你的 processor 是否会影响 argmax；还需重写 `AdapterLogitsProcessor.new_req_logits_processor(self,params)`，通过 `SamplingParams` 创建一个请求级 logits processor 实例：

??? code "如何包装请求级 logits processor 示例"

    ``` python
    ...

    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor, # 封装基类
        RequestLogitsProcessor, # 请求级 logits processor 类型
    )

    ...

    # 你的请求级 logits processor 示例
    class DummyPerReqLogitsProcessor:
        """请求级 logits processor，仅保留 target_token 的 logits"""

        def __init__(self, target_token: int) -> None:
            """指定 target_token"""
            self.target_token = target_token

        def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
        ) -> torch.Tensor:
            val_to_keep = logits[self.target_token].item()
            logits[:] = float("-inf")
            logits[self.target_token] = val_to_keep
            return logits

    ...

    # 封装请求级 logits processor 的示例
    class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
        """包装伪造的请求级 logits processor，生成 batch 级 logits processor"""

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(
            self,
            params: SamplingParams,
        ) -> Optional[RequestLogitsProcessor]:
            """为每个请求返回一个新的请求级 logits processor 实例。

            如果该请求不需要应用 logits processor，则返回 None。要求该请求必须有整数类型的 "target_token" 自定义参数。

            参数:
            params: 每个请求的采样参数

            返回:
            返回 Callable 形式的请求级 logits processor 或 None
            """
            target_token: Optional[Any] = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is None:
                return None
            if not isinstance(target_token, int):
                logger.warning(
                    "target_token value %s is not int; not applying logits"
                    " processor to request.",
                    target_token,
                )
                return None
            return DummyPerReqLogitsProcessor(target_token)
    ```

!!! note
    你可以在 `new_req_logits_processor()` 里返回 `None`，表示该请求不需要应用被包装的 logits processor。

当你实现了一个自定义子类（如 `WrappedPerReqLogitsProcessor`）包装你的请求级 logits processor 后，可以通过下文讲述的任一方式将该自定义类加载到 vLLM。

## 在 vLLM 加载自定义 Logits Processor 的方法

logits processor 需要在 vLLM 初始化时加载。注意，一旦 vLLM 引擎初始化完成，已加载的 logits processor 集合就不可再更改，不能按需为单个请求动态加载新 logits processor。

本节介绍让自定义 logits processor 被 vLLM 识别并加载的几种方式。

### 方法 1：在初始化时传入自定义 logits processor 的 FQCN（全限定类名）

此方法适用于 vLLM 离线和在线两种场景。自定义 logits processor 的 FQCN（即 `dotted.path.to.module:ClassName` 形式）可以作为参数传递给 `LLM` 和 `AsyncLLM` 的 Python 构造函数，或用命令行参数传递给 `vllm serve`，如下所示：

``` bash
vllm serve ... --logits_processors <logits processor 1> <logits processor 2> ...
```

FQCN 需满足以下条件：

1. Python 的 `importlib.import_module()` 能找到 FQCN 前半部分模块并加载

2. FQCN 后半部分的类名可从该模块导入

3. FQCN 指向的类必须是 `LogitsProcessor` 的子类

示例如下：

??? code "在 Python 中传递自定义 logits processor FQCN 给 `LLM`"

    ``` python
    # 传入 FQCN
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "在 Python 中传递自定义 logits processor FQCN 给 `AsyncLLM`"

    ``` python
    # 传入 FQCN
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=["your.module.path:DummyLogitsProcessor"])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

??? code "通过 CLI 给 vLLM server 传递自定义 logits processor FQCN"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

### 方法 2：自动检测 Python 环境中以 entry point 形式安装的自定义 logits processor

使用 [`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) 的 entry point 功能，可以让已安装的包自动作为插件提供给其它 Python 程序。

vLLM 初始化时会自动扫描 `vllm.logits_processors` entry point 分组，并加载所有通过 entry point 注册的 logits processor。

假设你开发了一个包含自定义 logits processor 的 Python 包，只需为每个 processor 配置一个 entrypoint。如在 `pyproject.toml` 文件中设置：

??? code "通过 Python entrypoint 暴露自定义 logits processor"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

包安装后，每次初始化 vLLM 时都会自动加载你的自定义 logits processor。通过 entrypoint 暴露的 processor 无需再显式传递给 `LLM`、`AsyncLLM` 构造函数或 vLLM server。

!!! note
    vLLM 会自动加载所有通过 `vllm.logits_processors` entry point 分组暴露的 logits processor。

### 方法 3（仅限离线）：将 Python 类对象直接传递给 vLLM 构造函数

你可以将自定义 logits processor 的类对象直接传递给 `LLM` 和 `AsyncLLM` 构造函数。此方法非常灵活，类对象既可以在当前源文件中定义，也可以从其他包导入。

??? code "在 Python 中传递自定义 logits processor 类对象给 `LLM` 或 `AsyncLLM`"

    ``` python
    # 从模块导入自定义 logits processor
    from some.module import DummyLogitsProcessor

    # ...或...

    # 本地定义自定义 logits processor
    from vllm.v1.sample.logits_processor import LogitsProcessor

    class DummyLogitsProcessor(LogitsProcessor):
        # 参考前文 DummyLogitsProcessor 实现
        ...

    # 将类对象传递给 LLM 构造函数
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[DummyLogitsProcessor],
    )

    # 传递类对象给 AsyncLLM 构造函数
    engine_args = AsyncEngineArgs(model="facebook/opt