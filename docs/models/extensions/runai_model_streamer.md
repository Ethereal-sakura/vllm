# 使用 Run:ai Model Streamer 加载模型

Run:ai Model Streamer 是一个支持并发读取张量（tensor）并将其流式传输到 GPU 内存的库。  
更多详细信息可参考 [Run:ai Model Streamer 官方文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md) 

vLLM 支持通过 Run:ai Model Streamer 加载 Safetensors 格式的权重文件。  
首先需要安装 vLLM 的 RunAI 可选依赖：

```bash
pip3 install vllm[runai]
```

如需以 OpenAI 兼容服务器方式运行，只需添加 `--load-format runai_streamer` 参数：

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer
```

从 AWS S3 对象存储加载模型，运行以下命令：

```bash
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

从 Google Cloud Storage 加载模型，运行以下命令：

```bash
vllm serve gs://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

如需从 S3 兼容对象存储加载模型，运行：

```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=true \
AWS_ENDPOINT_URL=https://storage.googleapis.com \
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

## 可调参数

可以通过 `--model-loader-extra-config` 调整参数：

`concurrency` 参数用于设置并发级别，即从文件读取张量到 CPU 缓冲区时的操作系统线程数量。  
对于从 S3 读取数据，此参数表示主机开启的 S3 客户端实例数量。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16}'
```

可以设置 CPU 内存缓冲区的大小限制，用于控制从文件读取到缓冲区的张量的内存占用。  
关于 CPU 缓冲区内存限制，可以参考 [相关说明](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit) 

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"memory_limit":5368709120}'
```

!!! note
    如需了解更多可调参数及通过环境变量配置的其他参数，请参考 [环境变量文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md) 

## 分片模型加载

vLLM 还支持通过 Run:ai Model Streamer 加载分片模型（sharded model）。这个功能非常适合将超大模型分成多个文件存储的场景。  
启用此功能，只需添加 `--load-format runai_streamer_sharded` 参数：

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

分片加载器要求模型文件遵循与常规分片状态加载器相同的命名规则：`model-rank-{rank}-part-{part}.safetensors`。  
你也可以通过在 `--model-loader-extra-config` 中设置 `pattern` 参数，自定义文件命名格式：

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"pattern":"custom-model-rank-{rank}-part-{part}.safetensors"}'
```

如果需要创建分片模型文件，可以使用 [examples/offline_inference/save_sharded_state.py](../../../examples/offline_inference/save_sharded_state.py) 脚本。此脚本演示了如何保存一个兼容 Run:ai Model Streamer 分片加载器的分片模型格式。

分片加载器支持所有常规 Run:ai Model Streamer 的可调参数，包括 `concurrency` 和 `memory_limit`，配置方式一致：

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

!!! note
    分片加载器在张量并行或流水线并行模型场景下表现尤为高效，因为每个 worker 只需读取自身所需的分片，而无需加载整个模型检查点文件。