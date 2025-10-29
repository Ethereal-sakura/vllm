# 插件系统

社区用户经常希望能为 vLLM 添加自定义功能。为此，vLLM 提供了插件系统，允许用户在不修改 vLLM 源码的前提下扩展新特性。本文将介绍 vLLM 插件的工作方式及如何为 vLLM 编写插件。

## vLLM 插件的运行机制

插件是由用户注册的代码，vLLM 会在运行时执行。由于 vLLM 的架构（详见 [架构总览](arch_overview.md)），在分布式推理和多种并行方式下，可能涉及多个进程。为确保插件能够生效，每个由 vLLM 创建的进程都需要加载插件。这一过程由 `vllm.plugins` 模块中的 [load_general_plugins](https://github.com/vllm-project/vllm/blob/c76ac49d266e27aa3fea84ef2df1f813d24c91c7/vllm/plugins/__init__.py#L16) 函数完成。每当 vLLM 启动新进程并准备开始工作时，都会调用该函数。

## vLLM 如何发现插件

vLLM 插件系统采用标准的 Python `entry_points` 机制。开发者可以通过该机制，将自定义函数注册到自己的 Python 包中，供其他包调用。以下是一个插件示例：

??? code

    ```python
    # 在 `setup.py` 文件中
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })

    # 在 `vllm_add_dummy_model.py` 文件中
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

关于如何为你的包添加 entry points，请参考 [官方文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) 

每个插件包含三个要素：

1. **插件组（Plugin group）**：即 entry point 分组的名称。vLLM 为通用插件使用 `vllm.general_plugins` 分组，该字段在 `setup.py` 的 `entry_points` 键中设置。为 vLLM 编写通用插件时，请始终使用 `vllm.general_plugins`。
2. **插件名称（Plugin name）**：即插件的名字，在 `entry_points` 字典的值中设置。上例中的插件名称为 `register_dummy_model`。可以通过设置环境变量 `VLLM_PLUGINS` 来筛选加载指定名称的插件。例如如果只想加载某个插件，则将 `VLLM_PLUGINS` 设为该插件名称。
3. **插件值（Plugin value）**：即需要注册到插件系统中的函数或模块的完整限定名。以上例中，插件值为 `vllm_add_dummy_model:register`，指的是 `vllm_add_dummy_model` 模块中的 `register` 函数。

## 支持的插件类型

- **通用插件**（分组名为 `vllm.general_plugins`）：主要用于将自定义的模型注册到 vLLM 中。通常是在插件函数内调用 `ModelRegistry.register_model` 完成模型注册。

- **平台插件**（分组名为 `vllm.platform_plugins`）：主要用于将自定义的平台注册到 vLLM。插件函数在当前环境不支持时应返回 `None`，支持时返回平台类的完整限定名。

- **IO 处理器插件**（分组名为 `vllm.io_processor_plugins`）：用于为池化模型注册自定义的前处理/后处理逻辑。插件函数返回 IOProcessor 类的完整限定名。

- **统计日志插件**（分组名为 `vllm.stat_logger_plugins`）：用于将自定义的日志记录器注册到 vLLM。入口点应为继承自 StatLoggerBase 的类。

## 编写插件的注意事项

- **可重入性**：entry point 指定的函数应具备可重入性，即可以多次调用而不会引发问题。这是因为在某些进程中，插件函数可能会被多次调用。

## 兼容性保障

vLLM 保证已文档化的插件接口（如 `ModelRegistry.register_model`）始终可用于插件注册模型。但插件开发者需自行确保插件与目标 vLLM 版本兼容。例如，`"vllm_add_dummy_model.my_llava:MyLlava"` 需要与插件适配的 vLLM 版本兼容。由于 vLLM 处于持续开发中，模型接口可能会发生变化。