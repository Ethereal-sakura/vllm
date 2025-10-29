# 节省内存

大型模型有可能导致你的设备内存不足（OOM）。下面介绍几种缓解这一问题的方法。

## 张量并行（Tensor Parallelism，TP）

张量并行（通过 `tensor_parallel_size` 参数设置）可以让模型在多张 GPU 间进行分布式拆分。

以下代码实现了将模型拆分到 2 张 GPU 上：

```python
from vllm import LLM

llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", tensor_parallel_size=2)
```

!!! warning
    为确保 vLLM 正确初始化 CUDA，请不要在初始化 vLLM 之前调用相关函数（如 [torch.cuda.set_device][]）  
    否则可能会遇到 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 这类错误。

    如果需要控制使用哪些设备，请通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来实现。

!!! note
    开启张量并行后，每个进程都会读取完整的模型并将其拆分为多个部分，因此模型的磁盘读取时间会变长（与张量并行的规模成正比）。

    你可以使用 [examples/offline_inference/save_sharded_state.py](../../examples/offline_inference/save_sharded_state.py) 将模型权重转换为分片（sharded）格式。虽然转换过程会花费一些时间，但之后加载分片权重会快很多。无论张量并行规模如何，加载模型的耗时都将保持稳定。

## 量化（Quantization）

量化后的模型可以显著减少内存占用，不过会牺牲一定的精度。

静态量化模型可以直接从 HF Hub 下载（部分常用模型可在 [Red Hat AI](https://huggingface.co/RedHatAI) 找到），无需额外配置即可使用。

同时也支持动态量化方式，通过 `quantization` 选项开启——详细用法见 [这里](../features/quantization/README.md)。

## 上下文长度与批大小

你还可以通过限制模型的上下文长度（`max_model_len` 参数）以及最大批处理数量（`max_num_seqs` 参数）来进一步降低内存使用。

```python
from vllm import LLM

llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
```

## 减少 CUDA 图（CUDA Graphs）占用

默认情况下，vLLM 通过 CUDA 图优化模型推理，这会在 GPU 上占用额外内存。

!!! warning
    CUDA 图捕获在 V1 版本中会比 V0 版本占用更多内存。

你可以调整 `compilation_config` 参数，在推理速度和内存占用之间做更好的平衡：

??? code

    ```python
    from vllm import LLM
    from vllm.config import CompilationConfig, CompilationMode

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            # 默认会按最大 batch 尺寸生成
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
    )
    ```

你也可以通过 `enforce_eager` 参数完全关闭 CUDA 图捕获：

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)
```

## 调整缓存大小

如果你的 CPU 内存不足，可以尝试以下方法：

- （仅限多模态模型）可以通过 `mm_processor_cache_gb` 引擎参数设置多模态缓存大小（默认 4 GiB）。
- （仅限 CPU 后端）可以通过设置 `VLLM_CPU_KVCACHE_SPACE` 环境变量来调整 KV 缓存大小（默认 4 GiB）。

## 多模态输入限制

你可以通过限制每个提示中多模态内容的数量，从而减少模型的内存占用：

```python
from vllm import LLM

# 每个提示最多接受 3 张图片和 1 个视频
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 3, "video": 1},
)
```

你也可以进一步，通过将未使用的模态内容数量设为零，彻底关闭不需要的模态。例如，如果你的应用只需图片输入，则无需为视频分配内存。

```python
from vllm import LLM

# 图片数量不限，但不接受视频
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"video": 0},
)
```

甚至可以只用多模态模型跑纯文本推理：

```python
from vllm import LLM

# 不接受图片，仅文本输入
llm = LLM(
    model="google/gemma-3-27b-it",
    limit_mm_per_prompt={"image": 0},
)
```

### 可配置选项

`limit_mm_per_prompt` 还支持按模态提供更细致的可配置选项。在可配置模式下，你依然需要填写 `count`，并可以选择性设置尺寸等参数，这样 vLLM 会根据你的实际媒体需求来预估和分配内存，而不是按照模型的极限最大值来分配。

不同模态可配置的参数如下：

- `image`：`{"count": int, "width": int, "height": int}`
- `video`：`{"count": int, "num_frames": int, "width": int, "height": int}`
- `audio`：`{"count": int, "length": int}`

详细说明可查阅 [`ImageDummyOptions`][vllm.config.multimodal.ImageDummyOptions]、[`VideoDummyOptions`][vllm.config.multimodal.VideoDummyOptions] 和 [`AudioDummyOptions`][vllm.config.multimodal.AudioDummyOptions]。

举例如下：

```python
from vllm import LLM

# 每个提示最多 5 张图片，按 512x512 尺寸预估
# 每个提示最多 1 个视频，按 32 帧 640x640 尺寸预估
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={
        "image": {"count": 5, "width": 512, "height": 512},
        "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640},
    },
)
```

为兼容旧版本，传递一个整数的写法依然有效，并等价于 `{"count": <int>}`。例如：

- `limit_mm_per_prompt={"image": 5}` 等同于 `limit_mm_per_prompt={"image": {"count": 5}}`
- 你可以混合使用不同格式：`limit_mm_per_prompt={"image": 5, "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640}}`

!!! note
    - 这些尺寸参数只影响内存预估，用于生成虚拟输入计算模型的激活内存预留，不会影响实际推理时输入的处理方式。
    - 如果你设置的参数超过模型支持的最大值，vLLM 会自动将其限制在模型允许的范围内，并可能输出警告日志。

!!! warning
    目前这些尺寸参数只影响激活内存的预估。编码器缓存（encoder cache）的大小取决于实际推理时的输入内容，不受这些参数限制。

## 多模态处理器参数

对于某些模型，你可以通过调整多模态处理器参数，进一步缩小处理后多模态输入的尺寸，从而节省内存。

例如：

```python
from vllm import LLM

# Qwen2-VL 系列模型可用
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_kwargs={"max_pixels": 768 * 768},  # 默认值为 1280 * 28 * 28
)

# InternVL 系列模型可用
llm = LLM(
    model="OpenGVLab/InternVL2-2B",
    mm_processor_kwargs={"max_dynamic_patch": 4},  # 默认值为 12
)
```