# 故障排查

本文档汇总了一些常见的故障排查策略。如果你怀疑遇到了 bug，请先[搜索已有 Issue](https://github.com/vllm-project/vllm/issues?q=is%3Aissue) ，看看是否已经被报告过。如果没有，请[新建 Issue](https://github.com/vllm-project/vllm/issues/new/choose)，并尽量提供详细的相关信息。

!!! note
    问题排查完成后，记得关闭所有已设置的调试环境变量，或者直接开启一个新的 shell，避免遗留的调试设置影响后续使用。否则系统可能因为开启调试功能而变慢。

## 下载模型时卡住

如果模型还没有下载到本地磁盘，vLLM 会自动从互联网下载模型，下载速度会受到网络状况影响。建议你先使用 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 预先下载模型，然后将本地路径传递给 vLLM，这样可以更好地定位问题。

## 从磁盘加载模型时卡住

如果模型体积较大，从磁盘加载可能需要较长时间。注意模型的存储位置，一些集群环境会使用跨节点共享文件系统（如分布式文件系统或网络文件系统），这类存储通常较慢。建议优先将模型存放在本地磁盘。此外，还要关注 CPU 内存占用情况，如果模型过大，可能会占用大量 CPU 内存，导致操作系统频繁进行磁盘与内存之间的交换，从而拖慢系统性能。

!!! note
    为了进一步定位模型下载和加载问题，可以使用 `--load-format dummy` 参数跳过加载模型权重。这样可以判断模型下载或加载过程是否是性能瓶颈。

## 内存不足

如果模型太大，无法完全加载到单块 GPU 上，就会出现 OOM（out-of-memory，内存溢出）报错。你可以参考[这些配置选项](../configuration/conserving_memory.md)来降低内存消耗。

## 文本生成质量变化

在 v0.8.0 版本中，[默认采样参数的来源发生了变化](https://github.com/vllm-project/vllm/pull/12622)。v0.8.0 之前，默认采样参数采用 vLLM 的中性默认值。从 v0.8.0 起，默认采样参数取自模型作者提供的 `generation_config.json` 文件。

大多数情况下，这能带来更高质量的输出，因为模型作者通常更了解自己的模型适合哪些采样参数。但也可能部分模型作者设定的默认值反而导致效果变差。

你可以通过尝试使用旧版默认参数进行对比：在线模式添加 `--generation-config vllm`，离线模式使用 `generation_config="vllm"`。如果这样做后生成效果有所提升，建议继续使用 vLLM 的默认参数，并向模型作者在 <https://huggingface.co> 提交反馈，建议他们优化 `generation_config.json` 的默认设置，以提升生成质量。

## 开启更详细的日志

如果以上方法都无法解决问题，vLLM 进程可能卡在某个环节。你可以通过以下环境变量开启更多调试日志，帮助定位问题：

- `export VLLM_LOGGING_LEVEL=DEBUG` ：开启详细日志输出。
- `export VLLM_LOG_STATS_INTERVAL=1.` ：更高频率输出运行队列、等待队列及缓存命中状态等统计信息。
- `export CUDA_LAUNCH_BLOCKING=1` ：排查具体是哪个 CUDA kernel 出现问题。
- `export NCCL_DEBUG=TRACE` ：开启 NCCL 详细日志。
- `export VLLM_TRACE_FUNCTION=1` ：记录所有函数调用，便于日志定位崩溃或卡死的具体函数。（警告：此选项会严重拖慢生成速度，超100倍，仅在必须时使用。）

## 断点调试

vLLM 的部分代码运行在子进程中，普通的 `pdb` 断点可能无法如预期生效，常见现象如下：

``` text
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 100, in trace_dispatch
    return self.dispatch_line(frame)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 125, in dispatch_line
    if self.quitting: raise BdbQuit
                      ^^^^^^^^^^^^^
bdb.BdbQuit
```

一种解决办法是使用 [forked-pdb](https://github.com/Lightning-AI/forked-pdb)。通过 `pip install fpdb` 安装，然后用如下方式设置断点：

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

另一种方式是完全关闭多进程，设置 `VLLM_ENABLE_V1_MULTIPROCESSING` 环境变量，这样调度器会在主进程中运行，就可以直接用标准的 `pdb` 断点了：

``` python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

## 网络配置问题

如果你的网络配置较为复杂，vLLM 实例可能无法正确获取本机 IP。你可以在日志中查找类似 `DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` 的信息，确保 IP 地址是正确的。如果不是，请用环境变量手动指定 IP：`export VLLM_HOST_IP=<your_ip_address>`。

此外，也可能需要设置 `export NCCL_SOCKET_IFNAME=<your_network_interface>` 和 `export GLOO_SOCKET_IFNAME=<your_network_interface>` ，指定 IP 所对应的网络接口。

## `self.graph.replay()` 附近报错

如果 vLLM 崩溃时，错误堆栈中出现在 `vllm/worker/model_runner.py` 的 `self.graph.replay()` 附近，这通常是 CUDAGraph 内部的 CUDA 错误。你可以在命令行加上 `--enforce-eager` 参数，或者在 [LLM][vllm.LLM] 类中设置 `enforce_eager=True`，以关闭 CUDAGraph 优化，从而进一步定位具体的 CUDA 操作问题。

## 硬件或驱动问题

如果 GPU/CPU 间通信无法正常建立，可以用下面的 Python 脚本进行检查，并按提示操作，确认 GPU/CPU 通信是否正常。

??? code

    ```python
    # 测试 PyTorch NCCL
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    data = torch.FloatTensor([1,] * 128).to("cuda")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    value = data.mean().item()
    world_size = dist.get_world_size()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch NCCL is successful!")

    # 测试 PyTorch GLOO
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    cpu_data = torch.FloatTensor([1,] * 128)
    dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
    value = cpu_data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch GLOO is successful!")

    if world_size <= 1:
        exit()

    # 测试 vLLM NCCL，含 cuda graph
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
    # pynccl 在 0.6.5+ 版本默认启用，
    # 0.6.4 及更低版本需手动启用。
    # 为兼容旧版本，保留此代码。
    pynccl.disabled = False

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        data.fill_(1)
        out = pynccl.all_reduce(data, stream=s)
        value = out.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL is successful!")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=g, stream=s):
        out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

    data.fill_(1)
    g.replay()
    torch.cuda.current_stream().synchronize()
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL with cuda graph is successful!")

    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()
    ```

如果你在单机环境测试，可以调整 `--nproc-per-node` 参数为你想用的 GPU 数量：

```bash
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<number-of-GPUs> test.py
```

多节点环境下，请根据实际情况调整 `--nproc-per-node` 和 `--nnodes`，并确保 `MASTER_ADDR` 设置为集群主节点的可访问 IP。运行如下命令：

```bash
NCCL_DEBUG=TRACE torchrun --nnodes 2 \
    --nproc-per-node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR test.py
```

如果脚本正常运行，会看到 `sanity check is successful!` 的提示。

如果脚本卡住或崩溃，通常说明硬件或驱动存在问题，建议联系系统管理员或硬件厂商进一步排查。常见的临时措施可以尝试设置 NCCL 相关环境变量，比如 `export NCCL_P2P_DISABLE=1`，有时能缓解问题。更多信息请查阅 [官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 。不过这些设置只建议临时使用，可能会影响系统性能。最佳方案还是修复硬件或驱动，使脚本可以正常运行。

!!! note
    多节点环境配置比单节点更为复杂。如果遇到类似 `torch.distributed.DistNetworkError` 的报错，往往是网络或 DNS 配置有误。此时可以手动分配节点 rank，并通过命令行参数指定 IP：

    - 在第一个节点运行：`NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 0 --master_addr $MASTER_ADDR test.py`
    - 在第二个节点运行：`NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 1 --master_addr $MASTER_ADDR test.py`

    请根据实际环境调整 `--nproc-per-node`、`--nnodes` 和 `--node-rank` 参数，并确保每个节点执行的命令（及 `--node-rank`）不同。

## Python 多进程

### `RuntimeError` 异常

如果日志中出现如下警告：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

或者出现类似如下 Python 报错：

??? console "日志"

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

此时你需要将所有对 `vllm` 的调用放在 `if __name__ == '__main__':` 判断内部。比如不要这样写：

```python
import vllm

llm = vllm.LLM(...)
```

而应该这样写：

```python
if __name__ == '__main__':
    import vllm

    llm = vllm.LLM(...)
```

## `torch.compile` 报错

vLLM 严重依赖 `torch.compile` 优化模型性能，因此需要 `torch.compile` 功能和 `triton` 库支持。默认情况下，vLLM 会利用 `torch.compile` [优化部分函数](https://github.com/vllm-project/vllm/pull/10406)。你可以在运行 vLLM 前，通过以下脚本验证 `torch.compile` 是否可用：

??? code

    ```python
    import torch

    @torch.compile
    def f(x):
        # 一个用来测试 torch.compile 的简单函数
        x = x + 1
        x = x * 2
        x = x.sin()
        return x

    x = torch.randn(4, 4).cuda()
    print(f(x))
    ```

如果报错信息中包含 `torch/_inductor` 相关内容，通常说明你安装的 `triton` 版本与当前 PyTorch 不兼容。可参考 <https://github.com/vllm-project/vllm/issues/12219> 进一步排查。

## 模型检查失败

如果出现如下报错：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] failed to be inspected. Please check the logs for more details.
```

说明 vLLM 导入模型文件失败。通常与依赖缺失或 vLLM 编译的二进制版本过老有关。请仔细阅读日志，定位具体报错原因。

## 模型不受支持

如果出现如下报错：

```text
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' object is not iterable
```

或：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] are not supported for now. Supported architectures: [...]
```

但你确定该模型已经列在[支持的模型列表](../models/supported_models.md)里，说明 vLLM 的模型识别可能存在问题。此时建议参考[这里的步骤](../configuration/model_resolution.md)手动指定 vLLM 对该模型的实现方式。

## 设备类型推断失败

如果看到类似 `RuntimeError: Failed to infer device type` 的报错，说明 vLLM 无法判断当前环境的设备类型。你可以查看[相关代码](../../vllm/platforms/__init__.py) ，了解 vLLM 是如何推断设备类型的，并分析为何没有按预期工作。自 [这个 PR](https://github.com/vllm-project/vllm/pull/14195) 后，也可以设置环境变量 `VLLM_LOGGING_LEVEL=DEBUG` 输出更详细日志，辅助排查问题。

## NCCL error: unhandled system error during `ncclCommInitRank`

如果你的服务场景采用 GPUDirect RDMA 跨节点分布式部署，且在 `ncclCommInitRank` 阶段遇到报错，哪怕设置了 `NCCL_DEBUG=INFO` 依然没有明确错误信息，日志大致如下：

```text
Error executing method 'init_device'. This might cause deadlock in distributed execution.
Traceback (most recent call last):
...
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl.py", line 99, in __init__
     self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 277, in ncclCommInitRank
     self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm),
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 256, in NCCL_CHECK
     raise RuntimeError(f"NCCL error: {error_str}")
 RuntimeError: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details)
...
```

这说明 vLLM 初始化 NCCL 通信器失败，常见原因包括缺少 `IPC_LOCK` Linux 权限或 `/dev/shm` 目录没有挂载。请参考 [启用 GPUDirect RDMA 的官方说明](../serving/parallelism_scaling.md#enabling-gpudirect-rdma) ，确保环境配置正确。

## 已知问题

- 在 `v0.5.2`、`v0.5.3` 和 `v0.5.3.post1` 版本