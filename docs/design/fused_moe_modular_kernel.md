# Fused MoE 模块化内核

## 简介

FusedMoEModularKernel 的实现详见 [此处](../..//vllm/model_executor/layers/fused_moe/modular_kernel.py)

根据输入激活（activation）的格式，FusedMoE 的实现大致可以分为两类：

* 连续 / 标准 / 非批处理（Non-Batched）
* 批处理（Batched）

!!! note
    文档中“连续”、“标准”和“非批处理”这几个术语是可以互换使用的。

输入激活的格式完全取决于所用的 All2All Dispatch。

* 连续模式下，All2All Dispatch 会返回一个形状为 (M, K) 的连续张量，以及形状为 (M, num_topk) 的 TopK Ids 和 TopK 权重。可以参考 `DeepEPHTPrepareAndFinalize` 的实现。
* 批处理模式下，All2All Dispatch 返回的激活张量形状为 (num_experts, max_tokens, K)。此时，属于同一个专家（expert）的激活/Token 会被聚集在一起。注意，并不是这个张量中的所有元素都是有效的。通常还会有一个 `expert_num_tokens` 张量，大小为 `num_experts`，其中 `expert_num_tokens[i]` 表示第 i 个专家实际包含的有效 token 数量。可以参考 `PplxPrepareAndFinalize` 或 `DeepEPLLPrepareAndFinalize` 的实现。

无论是连续还是批处理模式，FusedMoE 操作一般都包含多个步骤，下面的流程图对此进行了说明：

![](../assets/design/fused_moe_modular_kernel/fused_moe_non_batched.png "FusedMoE 非批处理")

![](../assets/design/fused_moe_modular_kernel/fused_moe_batched.png "FusedMoE 批处理")

!!! note
    批处理与非批处理在操作上的主要区别在于 Permute / Unpermute（置换/还原）操作，其余步骤基本一致。

## 设计动机

从上面的流程图可以看出，FusedMoE 涉及的操作非常多，并且每一步都可能有多种实现方式。不同操作组合起来，FusedMoE 的实现方式会迅速变得复杂难以管理。模块化内核框架正是为了解决这个问题，它将各个操作归纳为逻辑组件。这样的分类既便于管理，也能避免代码重复。同时，模块化内核将 All2All 的 Dispatch（分发）和 Combine（合并）实现与 FusedMoE 的实现解耦，可以分别独立开发和测试。此外，模块化内核还为各个组件引入了抽象类，为后续扩展提供了清晰的骨架。

后续文档将主要介绍连续 / 非批处理场景。扩展到批处理场景会很直接。

## 模块化内核组件

FusedMoEModularKernel 将 FusedMoE 操作拆分为三部分：

1. TopKWeightAndReduce
2. FusedMoEPrepareAndFinalize
3. FusedMoEPermuteExpertsUnpermute

### TopKWeightAndReduce

TopK 权重应用和归约（Reduce）步骤紧跟在 Unpermute 操作之后，All2All Combine 之前。需要注意的是，`FusedMoEPermuteExpertsUnpermute` 负责 Unpermute，`FusedMoEPrepareAndFinalize` 负责 All2All Combine。将 TopK 权重应用和归约放在 `FusedMoEPermuteExpertsUnpermute` 内是有意义的，但有些实现选择在 `FusedMoEPrepareAndFinalize` 环节进行。为了实现灵活配置，我们设计了 TopKWeightAndReduce 抽象类。

TopKWeightAndReduce 的具体实现见 [此处](../../vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py)。

`FusedMoEPrepareAndFinalize::finalize()` 方法接受一个 `TopKWeightAndReduce` 参数，在方法内部进行调用。
`FusedMoEModularKernel` 充当了 `FusedMoEPermuteExpertsUnpermute` 与 `FusedMoEPrepareAndFinalize` 的桥梁，决定 TopK 权重应用和归约的具体执行位置。

* 如果 `FusedMoEPermuteExpertsUnpermute` 实现本身包含权重应用和归约逻辑，则 `FusedMoEPermuteExpertsUnpermute::finalize_weight_and_reduce_impl` 方法返回 `TopKWeightAndReduceNoOp`。
* 如果需要在 `FusedMoEPrepareAndFinalize::finalize()` 执行权重应用和归约，则 `FusedMoEPermuteExpertsUnpermute::finalize_weight_and_reduce_impl` 方法会返回 `TopKWeightAndReduceContiguous` / `TopKWeightAndReduceNaiveBatched` / `TopKWeightAndReduceDelegate`。

### FusedMoEPrepareAndFinalize

`FusedMoEPrepareAndFinalize` 抽象类提供了 `prepare`、`prepare_no_receive` 和 `finalize` 方法。
`prepare` 负责输入激活的量化（Quantization）和 All2All 分发。`prepare_no_receive`（如果实现了）与 `prepare` 类似，不同的是它不会等待来自其他 worker 的结果，而是返回一个“receiver”回调，后续需要调用该回调以获得最终结果。不是所有 `FusedMoEPrepareAndFinalize` 类都必须实现该方法，如果实现了，可以用于和初始 All2All 通信过程并行处理其他工作，比如将共享专家和 Fused 专家交错处理。`finalize` 负责 All2All 合并，同时也可能包含 TopK 权重应用和归约（具体可参考 TopKWeightAndReduce 部分）。

![](../assets/design/fused_moe_modular_kernel/prepare_and_finalize_blocks.png "FusedMoEPrepareAndFinalize 组件示意图")

### FusedMoEPermuteExpertsUnpermute

`FusedMoEPermuteExpertsUnpermute` 类是 MoE 操作的核心。该抽象类主要包括以下几个方法：

* apply()
* workspace_shapes()
* finalize_weight_and_reduce_impl()

#### apply()

apply 方法主要实现以下操作：

* Permute（置换）
* 与权重 W1 做矩阵乘法（Matmul）
* 激活函数 + 乘法
* 量化（Quantization）
* 与权重 W2 做矩阵乘法（Matmul）
* Unpermute（还原）
* 可选：TopK 权重应用及归约

#### workspace_shapes()

FusedMoE 的核心实现会执行一系列操作。如果每步都单独为输出分配内存会造成效率低下。因此各个实现需要通过 workspace_shapes() 方法声明两个工作区的形状：工作区数据类型，以及 FusedMoE 输出的形状。`FusedMoEModularKernel::forward()` 方法会用这些信息分配工作区张量和输出张量，并将其传递给 `FusedMoEPermuteExpertsUnpermute::apply()` 方法。工作区可以作为中间缓冲区使用。

#### finalize_weight_and_reduce_impl()

有时候，在 `FusedMoEPermuteExpertsUnpermute::apply()` 内部直接进行 TopK 权重应用和归约会更高效。可以参考 [这个例子](https://github.com/vllm-project/vllm/pull/20228)。我们提供了 TopKWeightAndReduce 抽象类以支持这种实现方式，详见 TopKWeightAndReduce 部分。
`FusedMoEPermuteExpertsUnpermute::finalize_weight_and_reduce_impl()` 会返回一个 TopKWeightAndReduce 对象，供 `FusedMoEPrepareAndFinalize::finalize()` 使用。

![](../assets/design/fused_moe_modular_kernel/fused_experts_blocks.png "FusedMoEPermuteExpertsUnpermute 组件示意图")

### FusedMoEModularKernel

`FusedMoEModularKernel` 由 `FusedMoEPrepareAndFinalize` 和 `FusedMoEPermuteExpertsUnpermute` 两个对象组成。
伪代码如下：

```py
class FusedMoEModularKernel:
    def __init__(self,
                 prepare_finalize: FusedMoEPrepareAndFinalize,
                 fused_experts: FusedMoEPermuteExpertsUnpermute):

        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(self, DP_A):

        Aq, A_scale, _, _, _ = self.prepare_finalize.prepare(DP_A, ...)

        workspace13_shape, workspace2_shape, _, _ = self.fused_experts.workspace_shapes(...)

        # 分配工作区
        workspace_13 = torch.empty(workspace13_shape, ...)
        workspace_2 = torch.empty(workspace2_shape, ...)

        # 执行 fused_experts 操作
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace13, workspace2, ...)

        # 若 fused_experts 的实现已完成 TopK 权重应用和归约，则 war_impl 类型为 TopKWeightAndReduceNoOp
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()

        output = self.prepare_finalize.finalize(fe_out, war_impl,...)

        return output
```

## 实践指南

### 如何新增 FusedMoEPrepareAndFinalize 类型

一般来说，FusedMoEPrepareAndFinalize 类型是由某个 All2All 分发与合并的实现/内核作为底层支撑。例如：

* PplxPrepareAndFinalize 类型对应 Pplx All2All 内核，
* DeepEPHTPrepareAndFinalize 类型对应 DeepEP 高吞吐量 All2All 内核，
* DeepEPLLPrepareAndFinalize 类型对应 DeepEP 低延迟 All2All 内核。

#### 步骤 1：添加 All2All 管理器

All2All 管理器的作用是配置 All2All 内核实现。通常 `FusedMoEPrepareAndFinalize` 的实现会从 All2All 管理器获取内核实现的“句柄”，用于调用分发和合并方法。相关实现可参考 [此处](../../vllm/distributed/device_communicators/all2all.py)。

#### 步骤 2：添加 FusedMoEPrepareAndFinalize 类型

本节说明 `FusedMoEPrepareAndFinalize` 抽象类中各方法的意义。

`FusedMoEPrepareAndFinalize::prepare()`：实现量化和 All2All 分发。通常调用对应 All2All 管理器的分发方法。

`FusedMoEPrepareAndFinalize::has_prepare_no_receive()`：指示是否实现了 `prepare_no_receive`。默认为 False。

`FusedMoEPrepareAndFinalize::prepare_no_receive()`：实现量化和 All2All 分发，但不会等待结果，而是返回一个 thunk（可调用对象），后续手动等待最终结果。一般调用对应 All2All 管理器的分发方法。

`FusedMoEPrepareAndFinalize::finalize()`：可能包含 TopK 权重应用和归约，以及 All2All 合并。通常调用对应 All2All 管理器的合并方法。

`FusedMoEPrepareAndFinalize::activation_format()`：如果 prepare 方法（即 All2All 分发）的输出是批处理，则返回 `FusedMoEActivationFormat.BatchedExperts`，否则返回 `FusedMoEActivationFormat.Standard`。

`FusedMoEPrepareAndFinalize::topk_indices_dtype()`：TopK Ids 的数据类型。有些 All2All 内核对 TopK Ids 的数据类型有严格要求，会将该要求传递给 `FusedMoe::select_experts` 方法。如果没有特殊要求则返回 None。

`FusedMoEPrepareAndFinalize::max_num_tokens_per_rank()`：All2All 分发一次最多处理的 token 数量。

`FusedMoEPrepareAndFinalize::num_dispatchers()`：分发单元总数。该值决定了分发输出的形状，输出形状为 (num_local_experts, max_num_tokens, K)，其中 max_num_tokens = num_dispatchers() * max_num_tokens_per_rank()。

建议参考已有的 `FusedMoEPrepareAndFinalize` 实现，选择和你的 All2All 实现最接近的版本作为模板。

### 如何新增 FusedMoEPermuteExpertsUnpermute 类型

FusedMoEPermuteExpertsUnpermute 实现了 FusedMoE 的核心操作。其抽象类方法及意义如下：

`FusedMoEPermuteExpertsUnpermute::activation_formats()`：返回支持的输入与输出激活格式（连续/批处理）。

`FusedMoEPermuteExpertsUnpermute::supports_chunking()`：如果实现支持 chunking（分块处理），返回 True。一般输入为 `FusedMoEActivationFormat.Standard` 支持 chunking，`FusedMoEActivationFormat.BatchedExperts` 不支持。

`FusedMoEPermuteExpertsUnpermute::supports_expert_map()`：是否支持专家映射。

`FusedMoEPermuteExpertsUnpermute::workspace_shapes()` /
`FusedMoEPermuteExpertsUnpermute::finalize_weight_and_reduce_impl` /
`FusedMoEPermuteExpertsUnpermute::apply`：详见上方 “FusedMoEPermuteExpertsUnpermute” 部分。

### FusedMoEModularKernel 初始化流程

`FusedMoEMethodBase` 类有三个方法共同负责创建 `FusedMoEModularKernel` 对象：

* maybe_make_prepare_finalize
* select_gemm_impl
* init_prepare_finalize

#### maybe_make_prepare_finalize

该方法负责根据当前 all2all 后端的配置，构造合适的 `FusedMoEPrepareAndFinalize` 实例，比如在 EP + DP 场景启用时。基类方法目前会为 EP+DP 场景构造所有相关的 `FusedMoEPrepareAndFinalize` 对象。子类可以重写该方法以适配更多场景，比如 `ModelOptNvFp4FusedMoE` 可以为 EP+TP 场景构造 `FlashInferCutlassMoEPrepareAndFinalize`。
参考实现：

* `ModelOptNvFp4FusedMoE`

#### select_gemm_impl

该方法在基类中未定义，由子类负责实现，用于构造合适的 `FusedMoEPermuteExpertsUnpermute` 对象。
参考实现：

* `UnquantizedFusedMoEMethod`
* `CompressedTensorsW8A8Fp8MoEMethod`
* `CompressedTensorsW8A8Fp8MoECutlassMethod`
* `Fp8MoEMethod`
* `ModelOptNvFp4FusedMoE`
等子类。

#### init_prepare_finalize

该方法根据输入和环境变量，创建合适的 `FusedMoEPrepareAndFinalize` 对象。然后调用 `select_gemm_impl` 获取合适的 `FusedMoEPermuteExpertsUnpermute` 对象，最终构建出 `FusedMoEModularKernel` 对象。

请参考 [init_prepare_finalize](https://github.com/vllm-project/vllm/blob/1cbf951ba272c230823b947631065b826409fa62/vllm/model_executor/layers/fused_moe/layer.py#L188)。
**重要说明**：`FusedMoEMethodBase` 的子类会在其 `apply` 方法中使用 `FusedMoEMethodBase::fused_experts` 对象。当环境允许构造有效的 `FusedMoEModularKernel` 对象时，会用该对象覆盖 `FusedMoEMethodBase::fused_experts`，从而让子类无需关心具体采用哪种 FusedMoE 实现。

### 单元测试指南

我们为 `FusedMoEModularKernel` 提供了单元测试，详见 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py)。

测试会遍历所有 `FusedMoEPrepareAndFinalize` 和 `FusedMoEPremuteExpertsUnpermute` 类型的组合，如果兼容就会运行正确性测试。
如果你要添加新的 `FusedMoEPrepareAndFinalize` 或 `FusedMoEPermuteExpertsUnpermute` 实现：

1. 在 [mk_objects.py](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py) 中分别将新类型加入 `MK_ALL_PREPARE_FINALIZE_TYPES` 和 `MK_FUSED_EXPERT_TYPES`。
2. 在 [/tests/kernels/moe/modular_kernel_tools/common.py](../../tests/kernels/moe/modular_kernel_tools/common.py) 中更新 `Config::is_batched_prepare_finalize()`、`Config::is_batched_fused_experts()`、`Config::is_standard_fused_experts()`、`Config::is_fe_16bit_supported()`、`Config::is_fe_fp8_supported()`、`Config::is_fe_block_fp8_supported()`、`Config::is_fe_supports_chunking()` 方法。

按上述操作即可将新实现加入测试集。

### 如何检查 FusedMoEPrepareAndFinalize 与 FusedMoEPermuteExpertsUnpermute 的兼容性

单元测试文件 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) 可以作为独立脚本执行。
例如：`python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts`
这样可以检测两种类型的兼容性。如果类型不兼容，脚本会报错。

### 如何性能分析（Profile）

可参考 [profile_modular_kernel.py](../../tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py)
该脚本可为任意兼容的 `FusedMoEModularKernel::forward()` 调用生成 Torch trace。
例如：`python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel --pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts`

## FusedMoEPrepareAndFinalize