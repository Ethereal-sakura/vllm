# 注册模型

vLLM 依赖模型注册表（model registry）来决定如何运行每个模型。
你可以在[这里](../../models/supported_models.md)查看已预注册的模型架构列表。

如果你的模型不在这个列表中，你需要将其注册到 vLLM。
本页面将为你详细介绍注册流程。

## 内置模型

如果你想直接将模型加入 vLLM 库，首先需要 fork 我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)，然后按照[从源码构建](../../getting_started/installation/gpu.md#build-wheel-from-source)的说明进行安装。
这样你就可以自由修改代码库并测试你的模型。

在完成模型实现后（参考[教程](basic.md)），请将模型文件放到 [vllm/model_executor/models](../../../vllm/model_executor/models) 目录下。
接着，在 [vllm/model_executor/models/registry.py](../../../vllm/model_executor/models/registry.py) 文件中的 `_VLLM_MODELS` 列表里添加你的模型类，这样在导入 vLLM 时模型会自动注册。
最后，别忘了更新我们的[支持模型列表](../../models/supported_models.md)，让更多人了解你的模型！

!!! important
    每个部分中的模型列表都应按照字母顺序维护。

## 外部模型

你可以通过[插件机制](../../design/plugin_system.md)加载外部模型，无需更改 vLLM 代码库。

要注册模型，可以使用如下代码：

```python
# 你的插件入口文件
def register():
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

如果你的模型在导入时会初始化 CUDA，建议使用延迟导入（lazy-import），以避免出现 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 这样的错误：

```python
# 你的插件入口文件
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM",
    )
```

!!! important
    如果你的模型属于多模态模型，请确保模型类实现了 [SupportsMultiModal][vllm.model_executor.models.interfaces.SupportsMultiModal] 接口。
    详细介绍请查看[这里](multimodal.md)