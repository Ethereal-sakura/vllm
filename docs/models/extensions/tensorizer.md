# 使用 CoreWeave 的 Tensorizer 加载模型

vLLM 支持使用 [CoreWeave 的 Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 来加载模型  
将 vLLM 模型的张量（tensors）序列化到磁盘、HTTP/HTTPS 端点或 S3 端点后，可以在运行时直接将其高速反序列化到 GPU 上，大幅缩短 Pod 启动时间并减少 CPU 内存占用。同时还支持张量加密。

vLLM 已经将 Tensorizer 完全集成到其模型加载流程中。下面将简单介绍如何在 vLLM 上使用 Tensorizer。

## 安装 Tensorizer

要安装 `tensorizer`，只需运行 `pip install vllm[tensorizer]`。

## 基本用法

要通过 Tensorizer 加载模型，首先需要用 Tensorizer 对模型进行序列化  
可以参考[示例脚本](../../examples/others/tensorize_vllm_model.md)完成这个过程。

以序列化 `facebook/opt-125m` 为例，演示如何用脚本将其序列化并加载用于推理。

## 用 Tensorizer 序列化 vLLM 模型

要用 Tensorizer 序列化模型，只需在命令行通过示例脚本传递相应参数即可。该脚本的 docstring 详细说明了所有 CLI 参数和用法，这里直接采用其中的例子。假设你想将序列化后的模型保存到 S3 存储桶 `s3://my-bucket`：

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

这会将模型张量保存到 `s3://my-bucket/vllm/facebook/opt-125m/v1`。如果你还需要为序列化后的模型应用 LoRA 适配器，可以在上面的命令中指定 LoRA 适配器的 HF id，相应的文件也会一并保存：

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

## 使用 Tensorizer 部署模型

当模型序列化到指定位置后，就可以用 `vllm serve` 或 `LLM` 入口加载模型。只需将模型保存目录传给 `LLM()` 或 `vllm serve` 的 `model` 参数即可。例如，要加载之前带 LoRA 适配器的序列化模型，可以这样操作：

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora 
```

或者用 `LLM()`：

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
)
```

## Tensorizer 配置选项

`tensorizer` 的核心对象分别为 `TensorSerializer`（序列化）和 `TensorDeserializer`（反序列化）。如果你需要给这两个过程自定义参数，可以通过 `model_loader_extra_config` 传递，其中序列化参数用 `serialization_kwargs`，反序列化参数用 `deserialization_kwargs`。完整参数说明可参考 `tensorizer` 的 [serialization.py](https://github.com/coreweave/tensorizer/blob/main/tensorizer/serialization.py) 文件。

比如，在序列化时你可以通过 `TensorSerializer` 的 `limit_cpu_concurrency` 参数限制 CPU 并发数。要设置该参数，可以这样：

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --serialization-kwargs '{"limit_cpu_concurrency": 2}' \
   --suffix v1
```

如果你想在加载时自定义反序列化过程，比如通过 `TensorDeserializer` 的 `num_readers` 参数限制并发读取数量，可以像这样通过 `model_loader_extra_config` 传递：

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora \
    --model-loader-extra-config '{"deserialization_kwargs": {"num_readers": 2}}'
```

或者用 `LLM()`：

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
    model_loader_extra_config={"deserialization_kwargs": {"num_readers": 2}},
)
```