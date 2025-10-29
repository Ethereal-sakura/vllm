# Python 多进程（Multiprocessing）

## 调试

关于已知问题及其解决方法，请参考 [故障排查](../usage/troubleshooting.md#python-multiprocessing) 页面。

## 简介

!!! important
    文中的源码引用基于 2024 年 12 月时的代码状态。

在 vLLM 中使用 Python 多进程（multiprocessing）存在一定复杂性，主要原因包括：

- vLLM 作为一个库被调用时，无法完全控制使用它的外部代码
- 不同多进程启动方式与 vLLM 依赖项之间的兼容性各不相同

本文档将介绍 vLLM 是如何应对这些挑战的。

## 多进程启动方式

[Python 的多进程启动方式](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) 包括：

- `spawn` —— 启动一个新的 Python 进程。在 Windows 和 macOS 上为默认方式。

- `fork` —— 使用 `os.fork()` 复制 Python 解释器。在 Linux 上，Python 3.14 之前的默认方式。

- `forkserver` —— 启动一个服务器进程，按需 fork 新进程。Python 3.14 及以上版本在 Linux 上为默认方式。

### 各方式权衡

`fork` 是最快的方式，但与使用线程的依赖库不兼容。在 macOS 下使用 `fork` 可能导致进程崩溃。

`spawn` 与依赖库的兼容性更好，但当 vLLM 被作为库调用时可能会有问题。如果调用方代码没有加上 `__main__` 判断（`if __name__ == "__main__":`），vLLM 在 spawn 新进程时会意外地重复执行代码。这可能导致无限递归等各种问题。

`forkserver` 会启动一个服务器进程，根据需要 fork 新进程。不幸的是，当 vLLM 被当作库使用时，它也会遇到和 `spawn` 相同的问题。服务器进程其实也是以 spawn 方式创建的，如果没有加 `__main__` 判断，相关代码会被重复执行。

对于 `spawn` 和 `forkserver`，新进程不能依赖于继承全局状态，这与 `fork` 有很大不同。

## 依赖项兼容性

vLLM 的多个依赖明确提出“建议”或“要求”使用 `spawn`：

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

更准确地说，在这些依赖初始化后再用 `fork` 会有已知问题。

## 当前状态（v0）

可以通过环境变量 `VLLM_WORKER_MULTIPROC_METHOD` 控制 vLLM 使用哪种多进程方式。目前的默认值为 `fork`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

如果我们能确定进程是由 `vllm` 命令启动的（即我们拥有主进程），则优先选择 `spawn`，因为它兼容性最好。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor` 强制使用 `spawn`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

还有一些地方硬编码为 `spawn`：

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

相关 PR：

- <https://github.com/vllm-project/vllm/pull/8823>

## v1 版本中的历史行为

v1 引入了环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING`，用于控制 v1 引擎核心是否启用多进程，默认关闭。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

启用后，v1 的 `LLMEngine` 会新建一个进程来运行引擎核心。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

默认关闭多进程，正是因为上面提到的兼容性和作为库使用时的问题。

### v1 版本中的变动

Python 的 `multiprocessing` 没有一种放之四海而皆准的万能方案。我们的第一步是让 v1 能根据实际情况“尽力而为”地选择最佳多进程方式，以提升兼容性。

- 默认使用 `fork`
- 如果确定我们控制主进程（通过 `vllm` 启动），则使用 `spawn`
- 如果检测到 `cuda` 已经初始化，则强制使用 `spawn`，并给出警告。因为我们知道 `fork` 会出问题，这是目前最优的选择。

已知在这种情况下仍然会出错的场景：作为库使用 vLLM 且在调用 vLLM 前已初始化 `cuda`。我们会发出警告，提示用户加上 `__main__` 判断或关闭多进程功能。

如果遇到这个已知问题，用户会看到两条提示信息。首先是 vLLM 的日志：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

随后，Python 会抛出异常并给出详细说明：

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 其他方案的考量

### 检测是否有 `__main__` 判断

有建议认为，如果能检测到库调用方是否加了 `__main__` 判断，我们可以做得更好。这篇 [stackoverflow 讨论](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard) 也描述了类似问题。

虽然可以检测当前是否处于主进程或 spawn 出来的子进程，但并没有简单可靠的方式检测调用方代码是否加了 `__main__` 判断。

因此，这个方案被认为不切实际，已被放弃。

### 使用 `forkserver`

最初 `forkserver` 看起来是个不错的解决方案，但其实现机制决定了，只要 vLLM 作为库使用时，仍会遇到与 `spawn` 相同的问题。

### 始终强制使用 `spawn`

一种更彻底的做法是始终强制使用 `spawn`，并通过文档明确要求用户在作为库使用时必须加上 `__main__` 判断。但这样会破坏已有代码兼容性，也让 vLLM 更难用，不符合我们让 `LLM` 类易用的初衷。

因此，我们选择内部吸收复杂性，尽力让事情“自动好用”。

## 未来展望

未来我们可能会考虑采用不同的 worker 管理方式，绕开当前多进程带来的难题。

1. 可以实现类似 `forkserver` 的机制，但进程管理器由我们主动通过自定义子进程和专用入口（如 `vllm-manager` 进程）来启动和管理 worker。

2. 也可以调研其他更适合我们需求的第三方库。例如：

- <https://github.com/joblib/loky>
