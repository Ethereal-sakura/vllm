# 双批次重叠（Dual Batch Overlap）

## 动机

vLLM 中 DBO（Dual Batch Overlap）系统的核心目标是在 MoE（Mixture of Experts）层，将稀疏的全量通信过程与周边计算进行重叠加速。这个系统目前仅支持 DP+EP（数据并行+专家并行）部署。

## 简介

Dual Batch Overlap 系统的工作方式是：在模型执行器中将批次拆分成两份，启动两个工作线程，然后分别在这两个工作线程上运行模型。当启用 DBO 后，`FusedMoEModularKernel` 中的 yield 点允许两个 CPU 工作线程（也称为 UBatch 线程）进行“乒乓”切换：当一个线程正在计算时，另一个等待通信完成。代码中经常用 ubatch 简称 microbatch（微批次），这是 µ-batch 的 ASCII 友好写法。

DBO 系统对 `GpuModelRunner` 和 `ModularKernel` 进行了修改，并定义了两个工具类：`UBatchWrapper` 和 `UBatchContext`。`UBatchWrapper` 负责线程生命周期和 CUDA 图执行管理；`UBatchContext` 封装了 `ForwardContext`，用于协调两条 UBatch 线程之间的同步。

下面展示的是目前在 vLLM 中实现的重叠调度流程。

```python
# 调度符号说明：
#    S = 共享专家
#    A0 = MLA qkv 投影
#    A1 = 核心注意力 + 输出投影 + MoE 门控
#    D = 分发
#    C = 合并

# 计算: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# 通信: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# 顺序: D₁ 发送, A0₀, A1₀, D₁ 接收, D₀ 发送, MLP₁, D₀ 接收,
#        C₁ 发送, S₁, MLP₀, C₁ 接收, C₀ 发送, S₀, A0₁, A1₁, C₀ 接收
# MLP_SHARED_OVERLAP = "mlp_shared_overlap"
```

## 如何使用 DBO

启用 DBO 系统时，请在 vllm serve 命令中加入 `--enable-dbo` 参数。此功能需要同时设置 `--data-parallel-size N`（N 大于 1）和 `--enable-expert-parallel`。此外，还可以通过两个配置参数进行调整：

* `--dbo-decode-token-threshold`：对仅解码的批次，启用 DBO 所需的最少 Token 数
* `--dbo-prefill-token-threshold`：批次中至少有一个 prefill 时，启用 DBO 所需的最少 Token 数

目前 DBO 仅支持 DeepEP，因此必须预先安装 DeepEP。如果你的主要任务是解码请求，请将 `--all2all-backend` 设置为 `deepep_low_latency`；如果主要是 prefill 请求，则设置为 `deepep_high_throughput`。

以下是一个使用专家并行和 DBO 的两 DP rank 启动命令示例：
EX: `vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

请确保 `CUDA_VISIBLE_DEVICES` 中至少有两块 GPU 可见。

## DBO 组件

* GPUModelRunner
* UBatchWrapper
* UBatchContext

### GPU Model Runner

批次会由 `GPUModelRunner` 类拆分成微批次（microbatch）。主要分两步：首先，在所有 DP rank 间进行协调，判断是否可以启用微批次。微批处理必须在所有 DP rank 上一致。如果有任何 rank 无法进行微批处理，则所有 rank 都会禁用。如果可以进行微批处理，则所有 DP rank 的 Token 总数会补齐到最大值。如果某个 rank 补齐后第二个微批次为空，则微批处理会被终止。所有 rank 都确认拆分后，第二步就会执行：`GPUModelRunner` 将 `CommonAttentionMetadata` 一分为二，每个微批次对应一个 attention metadata。

### UBatchWrapper

gpu_ubatch_wrapper

`UBatchWrapper` 是模型封装类，负责 DBO 的线程管理、UBatchContext 管理和 CUDA 图的管理。它对 GPU Model Runner 尽量保持透明。

实现上会对模型执行两次，每次处理一个微批次。每次执行都在一个独立的 UBatch 线程中并行启动，通过 `UBatchContext` 进行同步。每个线程会获得对应的 attention metadata 切片，处理各自的批次数据。

CUDA 图由 `UBatchWrapper` 完全管理，所以 DBO 只支持全流程 CUDA 图模式。但一旦捕获了 DBO 的 CUDA 图，后续可直接复现，无需多线程或额外的 CPU 同步。

#### 接口说明

`__init__` 方法接收 model、VllmConfig、CUDAGraphMode 和 device。

`forward` 方法只接收模型参数。它会根据 `forward_context` 中是否存在 `ubatch_slices` 对象决定是否启用 DBO，否则按常规流程运行模型。

### UBatchContext

ubatch_context

`UBatchContext` 是一个 `ForwardContext` 的封装类，由 `UBatchWrapper` 用来同步两条 UBatch 线程。仅能通过 `make_ubatch_contexts` 进行实例化。

当某个 UBatch 线程执行到 `dbo_yield` 时会暂停，唤醒另一个线程，后者继续运行，直到也到达同样的 `dbo_yield`。这种“乒乓”机制会持续交换，直到整个模型运行完成。

目前所有的 `dbo_yield` 和 `dbo_maybe_run_recv_hook` 都在 `FusedMoEModularKernel.forward` 方法内调用。

#### 接口说明

`make_ubatch_context` 用于初始化两个 `UBatchContext`，分别对应两条 UBatch 线程。该接口接收两个 CUDA stream、已有的 `ForwardContexts` 和 CPU 线程屏障。建议仅通过此函数创建 `UBatchContext`，它会自动完成事件初始化。

`dbo_register_recv_hook` 方法用于注册回调，由另一条 UBatch 线程的 `UBatchContext` 中的 `FusedMoEPrepareAndFinalize` 类返回。另一线程调用 `dbo_maybe_run_recv_hook` 时会触发该回调。常用于等待全量通信内核完成。

`dbo_maybe_run_recv_hook` 方法会运行由 `dbo_register_recv_hook` 注册的回调（如存在）。

`dbo_yield` 方法负责让当前线程休眠，同时唤醒另一条 UBatch 线程。