# vLLM CLI 指南

vllm 命令行工具用于运行和管理 vLLM 模型。你可以通过以下命令查看帮助信息：

```bash
vllm --help
```

可用命令如下：

```bash
vllm {chat,complete,serve,bench,collect-env,run-batch}
```

## serve

启动 vLLM 的 OpenAI 兼容 API 服务端。

启动指定模型：

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

指定端口：

```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8100
```

通过 Unix 域套接字启动服务：

```bash
vllm serve meta-llama/Llama-2-7b-hf --uds /tmp/vllm.sock
```

使用 --help 获取更多选项：

```bash
# 列出所有参数分组
vllm serve --help=listgroup

# 查看某个参数分组
vllm serve --help=ModelConfig

# 查看某个单独参数
vllm serve --help=max-num-seqs

# 关键词搜索参数
vllm serve --help=max

# 分页查看完整帮助（less/more）
vllm serve --help=page
```

完整参数说明请参见 [vllm serve](./serve.md) 

## chat

通过已运行的 API 服务生成对话补全。

```bash
# 直接连接本地 API，无需参数
vllm chat

# 指定 API 地址
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 快速对话，只需一句提示语
vllm chat --quick "hi"
```

完整参数说明请参见 [vllm chat](./chat.md) 

## complete

通过已运行的 API 服务，根据给定提示生成文本补全。

```bash
# 直接连接本地 API，无需参数
vllm complete

# 指定 API 地址
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 快速补全，一句话提示
vllm complete --quick "The future of AI is"
```

完整参数说明请参见 [vllm complete](./complete.md) 

## bench

在线服务延迟和离线推理吞吐量基准测试。

如需使用基准测试相关命令，请通过 `pip install vllm[bench]` 安装额外依赖。

可用子命令如下：

```bash
vllm bench {latency, serve, throughput}
```

### latency

测试单批请求的延迟表现。

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

完整参数说明请参见 [vllm bench latency](./bench/latency.md) 

### serve

测试在线服务的吞吐量。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

完整参数说明请参见 [vllm bench serve](./bench/serve.md) 

### throughput

测试离线推理的吞吐量。

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

完整参数说明请参见 [vllm bench throughput](./bench/throughput.md) 

## collect-env

收集环境信息。

```bash
vllm collect-env
```

## run-batch

批量运行提示语，将结果写入文件。

本地文件运行示例：

```bash
vllm run-batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

远程文件运行示例：

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

完整参数说明请参见 [vllm run-batch](./run-batch.md) 

## 更多帮助

如需查看任意子命令的详细选项，可使用：

```bash
vllm <subcommand> --help
```
