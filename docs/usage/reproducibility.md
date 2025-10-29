# 可复现性

出于性能考虑，vLLM 默认并不保证结果的可复现性。如果你需要得到可复现的结果，请按以下方式设置：

- 对于 V1：关闭多进程，设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，让调度过程变为确定性（deterministic）。
- 对于 V0：设置全局种子（见下文）。

示例参考：[examples/offline_inference/reproducibility.py](../../examples/offline_inference/reproducibility.py)

!!! warning

    应用上述设置时，[会改变用户代码中的随机状态](#locality-of-random-state)。

!!! note

    即使采用了上述设置，vLLM 只有在相同的硬件和相同版本下才能保证结果的可复现性。
    同时，在线服务 API（`vllm serve`）不支持可复现性，因为在在线场景下几乎无法实现确定性的调度。

## 设置全局种子

vLLM 中的 `seed` 参数用于控制各种随机数生成器的随机状态。

如果你指定了 seed 的值，`random`、`np.random` 和 `torch.manual_seed` 的随机状态都会相应地设置。

但在某些情况下，设置种子也会[影响用户代码中的随机状态](#locality-of-random-state)。

### 默认行为

在 V0 中，`seed` 参数默认是 `None`。如果 `seed` 为 `None`，`random`、`np.random` 和 `torch.manual_seed` 的随机状态不会被设置。也就是说，每次运行 vLLM（若 `temperature > 0`）时，得到的结果都是不同的，这是预期行为。

在 V1 中，`seed` 参数默认是 `0`，这会为每个 worker 设置随机状态，所以即使 `temperature > 0`，每次运行 vLLM 得到的结果也会保持一致。

!!! note

    在 V1 中无法不指定 seed，因为不同的 worker 需要生成相同的输出，比如用于 speculative decoding 这类流程。
    
    详细信息见：<https://github.com/vllm-project/vllm/pull/17929>

### 随机状态的本地性

用户代码中的随机状态（即构造 [LLM][vllm.LLM] 类的代码）会在以下情况下被 vLLM 修改：

- 对于 V0：指定了 seed。
- 对于 V1：worker 和用户代码运行在同一进程下，比如设置了 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。

默认情况下，这些条件都不会生效，因此你可以放心使用 vLLM，不用担心影响到之后依赖随机状态的操作是否变为确定性。