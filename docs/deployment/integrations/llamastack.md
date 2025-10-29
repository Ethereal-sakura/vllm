# Llama Stack

vLLM 也可以通过 [Llama Stack](https://github.com/llamastack/llama-stack) 使用。

要安装 Llama Stack，请运行

```bash
pip install llama-stack -q
```

## 使用 OpenAI 兼容 API 进行推理

然后启动 Llama Stack 服务器，并按如下方式配置，使其连接到你的 vLLM 服务器：

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

更多关于远程 vLLM 提供商的详细说明，请参见 [这篇指南](https://llama-stack.readthedocs.io/en/latest/providers/inference/remote_vllm.html)

## 使用嵌入式 vLLM 进行推理

Llama Stack 还支持[内联（inline）推理提供商](https://github.com/llamastack/llama-stack/tree/main/llama_stack/providers/inline/inference)  
下面是使用该方式的配置示例：

```yaml
inference:
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
