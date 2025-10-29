# 架构概览

本文将为您介绍 vLLM 的整体架构。

[TOC]

## 入口点

vLLM 提供了多种与系统交互的入口。下图展示了它们之间的关系。

![Entrypoints Diagram](../assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM 类

LLM 类是进行离线推理（即不依赖独立推理服务器与模型交互）的主要 Python 接口。

以下是 `LLM` 类的使用示例：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 定义输入提示列表
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # 定义采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 使用 OPT-125M 模型初始化 LLM 引擎
    llm = LLM(model="facebook/opt-125m")

    # 生成每个输入提示的输出结果
    outputs = llm.generate(prompts, sampling_params)

    # 打印生成的结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

更多 API 细节请参考 API 文档的 [离线推理](../api/README.md#offline-inference) 部分。

`LLM` 类的实现代码位于 [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py)。

### OpenAI 兼容 API 服务器

另一个主要接口是 OpenAI 兼容的 API 服务器。你可以通过 `vllm serve` 命令启动该服务。

```bash
vllm serve <model>
```

`vllm` 命令行工具的代码位于 [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py)。

有时候你可能会直接使用 API 服务器的入口，而不是通过 `vllm` 命令。例如：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

!!! warning

    `python -m vllm.entrypoints.openai.api_server` 已不推荐使用
    ，未来版本可能将不再支持。

该代码实现位于 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py)。

API 服务器的更多细节可以参考 [OpenAI 兼容服务器](../serving/openai_compatible_server.md) 文档。

## LLM 引擎

`LLMEngine` 和 `AsyncLLMEngine` 两个类是 vLLM 系统的核心，负责模型推理和异步请求处理。

![LLMEngine Diagram](../assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine

`LLMEngine` 是 vLLM 引擎的核心组件，负责接收客户端请求，并通过模型生成响应结果。其功能涵盖输入处理、模型执行（可支持多主机和/或多 GPU 分布式部署）、调度及输出处理。

- **输入处理**：使用指定的分词器将输入文本进行分词处理。
- **调度**：决定每一步要处理哪些请求。
- **模型执行**：管理语言模型的实际推理，包括多 GPU 的分布式推理。
- **输出处理**：将模型生成的 token ID 解码为可读文本。

`LLMEngine` 的实现代码位于 [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py)。

### AsyncLLMEngine

`AsyncLLMEngine` 是 `LLMEngine` 的异步封装版本。它基于 `asyncio`，通过后台循环不断处理新到的请求。`AsyncLLMEngine` 适用于在线服务场景，可支持多并发请求，并将结果流式返回给客户端。

OpenAI 兼容 API 服务器采用了 `AsyncLLMEngine`。另外还有一个简单示例的 demo API 服务器，代码位于 [vllm/entrypoints/api_server.py](../../vllm/entrypoints/api_server.py)。

`AsyncLLMEngine` 的实现代码位于 [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py)。

## Worker

worker 是执行模型推理的进程。vLLM 遵循“一进程对应一个加速器设备（如 GPU）”的常规做法。例如，若采用 2 路张量并行（tensor parallelism）和 2 路流水线并行（pipeline parallelism），则总共有 4 个 worker。每个 worker 通过 `rank` 和 `local_rank` 标识，`rank` 用于全局调度，`local_rank` 主要用于分配本地加速器设备以及访问本地资源（如文件系统和共享内存）。

## Model Runner

每个 worker 内部都有一个 model runner 对象，负责模型的加载和运行。模型推理过程中的大部分逻辑都在这里实现，比如输入张量的准备和 cudagraph 的捕获。

## Model

每个 model runner 都包含一个 model 对象，也就是实际的 `torch.nn.Module` 实例。关于不同配置如何影响最终获得的类，可参见 [huggingface_integration](huggingface_integration.md)。

## 类层级结构

下图展示了 vLLM 的类层级关系：

> <figure markdown="span">
>   ![](../assets/design/hierarchy.png){ align="center" alt="query" width="100%" }
> </figure>

这个类层级背后有几个重要的设计考量：

1\. **可扩展性**：层级中的所有类都接收一个包含全部必要信息的配置对象。主配置对象为 [VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036)。由于类层级较深，每个类只需读取自己关心的配置项。将所有配置封装在一个对象中，便于在不同类之间传递和访问所需配置。例如，如果需要为模型运行器新增一个功能（这在 LLM 推理领域非常常见），只需在 `VllmConfig` 类中添加相应选项，模型运行器即可直接访问，无需修改 engine、worker、model 类的构造方法去传递新选项。

2\. **统一性**：模型运行器需要一个统一的接口来创建和初始化模型。vLLM 支持 50 多种主流开源模型，每个模型的初始化方式都可能不同。如果构造函数签名各有差异，模型运行器将难以统一调用，除非引入复杂且容易出错的逻辑。通过统一模型类的构造函数，模型运行器便可以在无需了解具体模型类型的情况下，轻松完成模型的创建和初始化。这对于模型组合也非常有用。例如，视觉-语言模型往往包含一个视觉模型和一个语言模型，构造函数统一后，可以方便地组合成一个整体模型。

!!! note
    为了实现这一目标，所有 vLLM 模型的构造函数已统一为：

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    构造函数现在只接受关键字参数，以避免传递错误参数。如果传递了旧格式配置，构造函数会直接报错。vLLM 开发者已在 vLLM 内部所有模型中完成了这一变更。对于自定义注册的模型，开发者需要自行适配，比如添加 shim 代码以兼容新旧构造方法：

    ??? code

        ```python
        class MyOldModel(nn.Module):
            def __init__(
                self,
                config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None,
                prefix: str = "",
            ) -> None:
                ...

        from vllm.config import VllmConfig
        class MyNewModel(MyOldModel):
            def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
                config = vllm_config.model_config.hf_config
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
                lora_config = vllm_config.lora_config
                super().__init__(config, cache_config, quant_config, lora_config, prefix)

        from packaging import version
        if version.parse(__version__) >= version.parse("0.6.4"):
            MyModel = MyNewModel
        else:
            MyModel = MyOldModel
        ```

    这样，模型即可同时兼容新旧版本的 vLLM。

3\. **初始化时的切分与量化**：某些功能需要修改模型权重，比如张量并行需要切分权重、量化需要对权重量化。实现这些功能有两种方式：一种是在模型初始化后再修改权重，另一种是在初始化期间直接处理。vLLM 选择了第二种方式。第一种方法对于大模型来说并不现实。例如，运行一个 405B 参数（约 810GB 权重）的模型在 16 张 H100 80GB 显卡上，理想情况下每张卡只需加载 50GB 权重。如果等模型初始化后再切分，所有 GPU 都要加载完整 810GB 权重，内存开销极大。而初始化时直接切分，则每一层只会创建所需的权重 shard，大大减少了内存占用。量化同理。注意，模型构造函数还增加了一个 `prefix` 参数，用于根据前缀区别初始化行为，这对于非均匀量化（模型不同部分采用不同量化方式）非常有用。通常顶层模型的 `prefix` 为空字符串，子模型如视觉、语言部分则会用 `"vision"`、`"language"` 等前缀。这与 checkpoint 文件中模块 state dict 的命名保持一致。

这种设计的一个不足是：vLLM 各个组件都需要完整的配置对象，导致难以对单个组件进行单元测试。为此，我们提供了默认初始化函数，生成所有字段为 `None` 的默认配置对象。如果被测试的组件只关心部分配置项，只需设置相关字段即可，从而支持组件的独立测试。需要注意的是，vLLM 的测试多为端到端整体测试，因此这一问题影响有限。

总之，完整配置对象 `VllmConfig` 可以视为引擎级别的全局状态，在所有 vLLM 类之间共享。