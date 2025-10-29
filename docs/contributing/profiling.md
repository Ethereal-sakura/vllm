# vLLM 性能分析指南

!!! warning
    性能分析（Profiling）仅适用于 vLLM 的开发者和维护者，用于了解代码库中各部分的耗时比例。**普通用户请勿开启性能分析功能**，因为它会极大降低推理速度。

## 使用 PyTorch Profiler 进行分析

我们支持通过 `torch.profiler` 模块对 vLLM worker 进行追踪。只需设置环境变量 `VLLM_TORCH_PROFILER_DIR` 为你希望保存追踪数据的目录，例如：`VLLM_TORCH_PROFILER_DIR=/mnt/traces/`。你还可以通过以下环境变量进一步控制分析内容：

- `VLLM_TORCH_PROFILER_RECORD_SHAPES=1`：记录张量形状，默认关闭
- `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1`：记录内存使用情况，默认关闭
- `VLLM_TORCH_PROFILER_WITH_STACK=1`：记录堆栈信息，默认开启
- `VLLM_TORCH_PROFILER_WITH_FLOPS=1`：记录 FLOPs（浮点运算量），默认关闭

OpenAI 服务端也需要设置环境变量 `VLLM_TORCH_PROFILER_DIR`。

使用 `vllm bench serve` 命令时，添加 `--profile` 参数即可开启性能分析。

数据追踪结果可以通过 <https://ui.perfetto.dev/> 进行可视化。

!!! tip
    你可以直接使用命令 `python -m vllm.entrypoints.cli.main bench` 调用 bench 模块，无需提前安装 vLLM。

!!! tip
    分析时只需发送少量请求，否则追踪文件会很大。追踪文件无需解压，可以直接查看。

!!! tip
    停止 profiler 时，系统会将所有追踪数据写入目录，这可能需要较长时间。例如处理 100 个请求的数据，Llama 70b 在 H100 上写入大约需 10 分钟。建议在启动服务前设置 `VLLM_RPC_TIMEOUT` 环境变量为较大值，如 30 分钟：
    `export VLLM_RPC_TIMEOUT=1800000`

### 命令示例及用法

#### 离线推理

更多参考见 [examples/offline_inference/simple_profiling.py](../../examples/offline_inference/simple_profiling.py)。

#### OpenAI 服务端

```bash
VLLM_TORCH_PROFILER_DIR=./vllm_profile \
    vllm serve meta-llama/Meta-Llama-3-70B
```

vllm bench 命令示例：

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Meta-Llama-3-70B \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

## 使用 NVIDIA Nsight Systems 进行分析

Nsight Systems 是一款高级性能分析工具，可以展示寄存器和共享内存使用情况、代码注释区块、底层 CUDA API 和事件等详细信息。

[安装 nsight-systems](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) 可使用包管理器。以下示例适用于 Ubuntu：

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

### 命令示例及用法

使用 `nsys` 进行分析时，建议设置环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。默认是 `fork`，但 `spawn` 更适合 Nsight。详细信息参考 [Nsight Systems 发布说明](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues)。

#### 离线推理

基本用法：在原有推理脚本前加上 `nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node` 即可。

以下为 `vllm bench latency` 脚本的示例：

```bash
nsys profile -o report.nsys-rep \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI 服务端

服务端分析时，也需在 `vllm serve` 命令前加上 `nsys profile`，并根据基准测试需求指定 `--delay XX --duration YY` 参数。达到指定持续时间后，服务器会自动关闭。

```bash
# 服务端
nsys profile -o report.nsys-rep \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --delay 30 \
    --duration 60 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct

# 客户端
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 1 \
    --dataset-name random \
    --random-input 1024 \
    --random-output 512
```

实际使用时，建议将 `--duration` 设置为较大值。需要停止分析时，可运行：

```bash
nsys sessions list
```

获取形如 `profile-XXXXX` 的 session id，然后运行：

```bash
nsys stop --session=profile-XXXXX
```

即可手动终止 profiler，并生成 `nsys-rep` 报告。

#### 分析方法

你可以通过 CLI 查看概要信息（如 `nsys stats [profile-file]`），或按照[官方指引](https://developer.nvidia.com/nsight-systems/get-started)在本地安装 Nsight GUI 查看详细报告。

??? console "CLI 示例"

    ```bash
    nsys stats report1.nsys-rep
    ...
    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

    Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                                                  Name
    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off…
        9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_…
        5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(T1 *, cons…
        4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa…
        2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type vllm::fused_add_rms_norm_kern…
        1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  void vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>(const long *, T1 *, T1 *, const T1 *, in…
        0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0…
    ...
    ```

GUI 示例：

<img width="1799" alt="Screenshot 2025-03-05 at 11 48 42 AM" src="https://github.com/user-attachments/assets/c7cff1ae-6d6f-477d-a342-bd13c4fc424c" />

## 持续性能分析

PyTorch 基础设施仓库的 [GitHub CI 工作流](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-profiling.yml) 提供了对 vLLM 不同模型的持续性能分析。自动分析可以帮助跟踪不同模型及配置的性能变化和优化机会。

### 工作原理

该工作流目前会定期（每周）对选定模型进行性能分析，生成详尽的性能追踪数据，可用多种工具进行分析，便于发现性能回退或优化空间。你也可以手动通过 Github Action 触发分析流程。

### 添加新模型

如需添加其他模型参与持续性能分析，只需修改 PyTorch integration testing 仓库中的 [profiling-tests.json](https://github.com/pytorch/pytorch-integration-testing/blob/main/vllm-profiling/cuda/profiling-tests.json) 配置文件，将你的模型信息补充进去即可。

### 查看分析结果

持续分析流程生成的追踪数据可在 [vLLM 性能仪表盘](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) 公共访问。查找 **Profiling traces** 表格即可浏览并下载不同模型的分析数据。

## vLLM Python 代码性能分析

Python 标准库自带 [cProfile](https://docs.python.org/3/library/profile.html) 性能分析工具。vLLM 提供了简便的辅助方法，可以直接应用于 vLLM 的代码区块分析。你可以用 `vllm.utils.profiling.cprofile` 或 `vllm.utils.profiling.cprofile_context` 对代码片段进行性能分析。

!!! note
    过时的导入路径 `vllm.utils.cprofile` 和 `vllm.utils.cprofile_context` 已不推荐使用。
    请改用 `vllm.utils.profiling.cprofile` 和 `vllm.utils.profiling.cprofile_context`。

### 用法示例 - 装饰器

第一个辅助工具是 Python 装饰器，可以直接分析某个函数。如果指定了文件名，分析结果会保存至该文件，否则会直接输出到标准输出。

```python
from vllm.utils.profiling import cprofile

@cprofile("expensive_function.prof")
def expensive_function():
    # 一些耗时的代码
    pass
```

### 用法示例 - 上下文管理器

第二个辅助工具是上下文管理器，可以分析一段代码区块。和装饰器类似，文件名参数可选。

```python
from vllm.utils.profiling import cprofile_context

def another_function():
    # 更耗时的代码
    pass

with cprofile_context("another_function.prof"):
    another_function()
```

### 分析结果

有多种工具可以帮助分析生成的性能报告。例如可以用 [snakeviz](https://jiffyclub.github.io/snakeviz/) 可视化分析结果：

```bash
pip install snakeviz
snakeviz expensive_function.prof
```

### 分析垃圾回收（GC）开销

可通过环境变量 VLLM_GC_DEBUG 调试垃圾回收开销：

- VLLM_GC_DEBUG=1：启用 GC 调试器，记录每次 gc.collect 的耗时
- VLLM_GC_DEBUG='{"top_objects":5}'：启用 GC 调试器，每次 gc.collect 时记录被回收最多的前 5 个对象