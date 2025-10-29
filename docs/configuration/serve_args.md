# Server Arguments

`vllm serve` 命令用于启动兼容 OpenAI 的服务端。

## 命令行参数

`vllm serve` 命令用于启动兼容 OpenAI 的服务端。
想要查看可用选项，请参考 [命令行参考](../cli/README.md#options)！

## 配置文件

你可以通过 [YAML](https://yaml.org/) 配置文件加载命令行参数。
参数名必须使用上面提到的长格式名称（详见 [上文](serve_args.md)）。

例如：

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

要使用上面的配置文件：

```bash
vllm serve --config config.yaml
```

!!! note
    如果某个参数同时在命令行和配置文件中提供，则命令行中的值优先。
    参数优先级顺序为：`命令行 > 配置文件 > 默认值`。
    例如：`vllm serve SOME_MODEL --config config.yaml`，此时 SOME_MODEL 优先于配置文件中的 `model`。