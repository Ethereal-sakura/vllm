# 自定义参数

你可以使用 vLLM 的*自定义参数（custom arguments）*，为 vLLM 传递那些不属于 vLLM `SamplingParams` 和 REST API 规范的参数。添加或删除 vLLM 自定义参数时，无需重新编译 vLLM，因为这些自定义参数以字典的形式传递。

自定义参数非常适合在不修改 vLLM 源码的情况下，比如你想使用[自定义 logits 处理器（custom logits processor）](./custom_logitsprocs.md)时使用。

## 离线自定义参数

通过 `SamplingParams.extra_args` 以 `dict` 形式传递的自定义参数，对所有能访问到 `SamplingParams` 的代码都是可见的：

``` python
SamplingParams(extra_args={"your_custom_arg_name": 67})
```

这样，你就可以把那些不属于 `SamplingParams` 的参数，作为请求的一部分传递给 `LLM`。

## 在线自定义参数

vLLM 的 REST API 支持通过 `vllm_xargs` 向 vLLM 服务器传递自定义参数。下面的例子展示了如何在 vLLM REST API 请求中集成自定义参数：

``` bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        ...
        "vllm_xargs": {"your_custom_arg": 67}
    }'
```

此外，使用 OpenAI SDK 时，也可以通过 `extra_body` 参数访问 `vllm_xargs`：

``` python
batch = await client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    ...,
    extra_body={
        "vllm_xargs": {
            "your_custom_arg": 67
        }
    }
)
```

!!! note
    `vllm_xargs` 实际上会被赋值给 `SamplingParams.extra_args`，因此只要你的代码用的是 `SamplingParams.extra_args`，就可以同时兼容离线和在线两种场景。