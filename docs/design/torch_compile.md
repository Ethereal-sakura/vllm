# `torch.compile` 集成

在 vLLM 的 V1 架构中，`torch.compile` 默认开启，并且是框架的核心部分。本文将通过一个简单的例子，帮助你理解 `torch.compile` 的使用方式。

在本例中，我们将运行一个常见的 Llama 模型，并将日志级别设置为 debug，以便查看详细信息。你可以使用如下命令：`VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`。

!!! note
    关于 `torch.compile` 集成的更多信息和最新进展，请参考这篇 [博客文章](https://blog.vllm.ai/2025/08/20/torch-compile.html)

## 编译缓存（Compilation Cache）

在详细日志中，你会看到如下内容：

```console
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM 会根据当前的各种因素，自动选择一个目录来存储所有编译产物。这意味着，在部署时你可以直接复制整个 `~/.cache/vllm/torch_compile_cache` 目录，从而大幅减少编译时间，加快 vLLM 实例的启动速度。

影响缓存目录的因素包括：

- 所有相关配置（参考 [config 文件夹](../../vllm/config)中各配置的 `compute_hash` 函数）
- PyTorch 的相关配置（参考 [compiler_interface.py](../../vllm/compilation/compiler_interface.py) 的 `compute_hash` 函数）
- 模型的 forward 函数及其调用的相关函数（见下文说明）

综合以上因素，通常可以保证缓存的安全性，不会引发异常行为。因此，缓存默认是开启的。如果你需要调试编译过程，或者怀疑缓存存在问题，可以通过设置环境变量 `VLLM_DISABLE_COMPILE_CACHE=1` 来禁用缓存。

vLLM 的 `torch.compile` 集成有一个独特之处：所有编译过程会在服务请求前全部完成，用户请求不会触发新的编译。否则，编译过程会阻塞请求，导致响应时间出现不可预期的延迟。

## Python 代码编译

在详细日志中，你可以看到：

??? console "日志"

      ```text
      DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

      DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

      DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
      DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
      ```

这部分日志展示的是 Python 代码编译过程，也就是通过 Dynamo 进行计算图捕获。它会追踪 `xxx/vllm/model_executor/models/llama.py:339` 这一行的 forward 函数。在 forward 过程中，Dynamo 还会内联并追踪其它相关函数，包括 PyTorch 的一些函数（比如 `xxx/torch/nn/modules/module.py`，因为访问 module 属性会调用函数）、以及 vLLM 自己的通信、注意力、激活等函数。所有被追踪到的文件都会被纳入缓存目录的计算中，这样只要这些文件有任何变动，都会导致缓存失效并重新编译。

Dynamo 编译的结果，是一个新的函数，存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py`，它一般负责从模块中解包张量，并传递给追踪到的计算图。而计算图则保存在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py`。

## 计算图处理

计算图对每一个张量都带有 shape 注解。输入包括 input ids、position ids、模型权重和 buffer，输出则是最终的 hidden states。需要注意的是，lm head projection 和采样操作并不会包含在计算图内。

计算图的大部分输入 shape 是静态的，因为它们是模型权重和 buffer，在模型生命周期内不会变化。只有 input ids 和 position ids 是符号 shape，也就是说它们的 shape 会随着每次 batch 而变化，不过它们会共享同一个符号 shape。换句话说，唯一会变化的其实是 batch size（即本轮 forward 处理的 token 数）。

注意力操作较为复杂，需要和 kv 缓存（kv caches）进行交互，且其 shape 也较为复杂。幸运的是，注意力操作的输出 shape 通常与输入 query 的 shape 相同。因此，我们将整个注意力操作封装为一个 PyTorch 自定义算子 `torch.ops.vllm.unified_attention_with_output`，这样 Dynamo 就不会去分析内部实现。借助这种方法，虽然注意力操作很复杂，但我们依然能在 Dynamo 视角下完整捕获模型的主计算图。

整个计算图还会被 `splitting_ops`（通常指 attention 操作）进一步切分。因此，在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 文件中，可以看到许多子模块，每个子模块都是 attention 切分后的一段计算图：

- 注意力操作本身是一个子模块
- 从一次注意力到下一次注意力之间的计算图，也是一个子模块

每个子模块都有独立的索引，并单独处理。

## 计算图编译

在详细日志中，还可以看到如下内容：

```console
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

这表明，第 0 段计算图（shape 为 `None`，即符号 shape）通过 Inductor 编译，key 为 `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`，生成的 kernel 文件位于 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py`。你可以打开这个文件查看 Inductor 最终运行的代码。

另外你会发现，第 1 段和第 15 段的 key 是一样的，而第 0 段和第 16 段是不同的。这是正常现象，因为我们以 attention op 为界切分计算图后，通常会得到 3 种独特的子图：

- attention 之前的第一个层
- 各个 attention 之间的中间层
- attention 之后的最后一层

如果缓存目录已经存在（比如第二次运行同样的代码），你会看到如下日志：

```console
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

此时，Inductor 编译过程会被完全跳过，直接从磁盘加载先前生成的编译产物。

上面例子中，Inductor 针对通用 shape（即符号 shape）进行编译。你也可以让 Inductor 针对特定 shape 编译，比如：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

这样会为 batch size 为 `1, 2, 4, 8` 时单独编译 kernel。此时，计算图中的所有 shape 都是静态且已知的，系统会自动开启性能调优（auto-tuning），以获得最佳性能。首次运行时会比较慢，但下次运行时可以直接复用缓存，无需再次调优。

当所有 shape 都已知时，`torch.compile` 会自动对比不同配置，通常能找到更优的 kernel 配置。例如，你会看到类似日志：

??? console "日志"

    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
      mm 0.0160 ms 81.6% 
      triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
      triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
      triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
      triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
    ```

这说明，对于 `8x2048x3072` 的矩阵乘法，`torch.compile` 尝试了多种 triton 配置模板，最终效果远超默认代码（默认会调度到 cublas 库）。

不过需要注意，自动调优过程会花费较长时间（从几秒到几分钟，取决于模型和 batch size），虽然结果可以缓存，考虑到易用性我们默认关闭 auto-tuning。如果你追求极致性能，建议针对特定 shape 进行编译和调优。

## Cudagraph 捕获

vLLM 的 V1 架构采用了分段式 cudagraph，与分段编译逻辑一致。整个计算图会被拆分成若干段，每一段 attention 之间的子图（包括最前面、最后面的部分）都会单独捕获 cudagraph。原因在于，attention 之间的计算通常是按 token 处理的，非常适合 cudagraph；而 attention 本身由于兼容性问题，不容易直接用 cudagraph。因此，我们让 attention 操作在 eager 模式下执行，其它部分则用 cudagraph，以此兼顾灵活性与性能。

分段式 cudagraph 还带来了更精细的内存管理。这样做的目的是，只将 attention kernel 排除在 cudagraph 之外，其余模块和所有内存分配操作都包含在 cudagraph 内。这也是为什么 V1 版本的 attention 操作输出张量要作为 attention 的输入之一。

这些 cudagraph 由编译后端负责捕获