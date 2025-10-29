# IO Processor 插件

IO Processor 插件是一项可以对池化模型（pooling models）的输入和输出进行前置和后置处理的功能。用户可以通过插件将自定义输入传递给 vLLM，插件会将这些输入转换为一个或多个模型提示（prompt），并送入模型的 `encode` 方法。这样可以实现例如多模态数据生成等高级用法，比如用户将一张图片输入 vLLM，并获得一张图片作为输出。

使用 IO Processor 插件进行推理时，输入提示的类型由插件定义，最终请求的输出类型也同样由插件决定。vLLM 不会对输入或输出数据进行任何校验，确保数据格式正确是插件的责任。当前，这类插件仅支持池化模型，并可通过 `LLM` 和 `AsyncLLM` 的 `encode` 方法调用，或者在在线服务模式下通过 `/pooling` 接口触发。

## 如何编写 IO Processor 插件

IO Processor 插件需要实现 [`IOProcessor`][vllm.plugins.io_processors.interface.IOProcessor] 接口：

```python
IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # 无法保证输出顺序与输入到 vLLM 的顺序一致
        # 在后处理之前按 id 排序
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> IOProcessorInput:
        raise NotImplementedError

    def validate_or_generate_params(
        self, params: SamplingParams | PoolingParams | None = None
    ) -> SamplingParams | PoolingParams:
        return params or PoolingParams()

    @abstractmethod
    def output_to_response(
        self, plugin_output: IOProcessorOutput
    ) -> IOProcessorResponse:
        raise NotImplementedError
```

`parse_request` 方法用于校验用户的输入提示，并将其转换为 `pre_process` / `pre_process_async` 方法所需的格式。
`pre_process*` 方法会使用已校验的插件输入，生成供 vLLM 正常推理用的模型提示。
`post_process*` 方法以 `PoolingRequestOutput` 对象为输入，生成自定义的插件输出。
`validate_or_generate_params` 方法用于校验用户请求中传递的 `SamplingParameters`（采样参数）或 `PoolingParameters`（池化参数），如果没有指定，则自动生成新的参数。该方法始终返回校验或生成后的参数。
`output_to_response` 方法仅用于在线服务，将插件输出转换为 `IOProcessorResponse` 类型，最终由 API Server 返回。`/pooling` 服务接口的实现可参考 [vllm/entrypoints/openai/serving_pooling.py](../../vllm/entrypoints/openai/serving_pooling.py)。

一个使用 PrithviGeospatialMAE 模型生成 geotiff 图片的插件实现示例可见 [这里](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation) 。
此外，可参考我们的在线示例（[examples/online_serving/prithvi_geospatial_mae.py](../../examples/online_serving/prithvi_geospatial_mae.py)）和离线推理示例（[examples/offline_inference/prithvi_geospatial_mae_io_processor.py](../../examples/offline_inference/prithvi_geospatial_mae_io_processor.py)）。

## 如何使用 IO Processor 插件

IO Processor 插件会在引擎启动时加载，指定插件名称有两种方式：

1. 通过 vLLM 的 `EngineArgs`：在初始化 `AsyncLLM` 时设置 `io_processor_plugin` 参数。离线模式下也可以直接将该参数传递给 `LLM`，在线服务模式下则通过 `--io-processor-plugin` 命令行参数指定。
2. 通过模型的 HF 配置：在模型配置（config.json）中添加 `io_processor_plugin` 字段。

优先级由设置顺序决定。也就是说，如果同时通过 `EngineArgs` 和模型 HF 配置指定了插件名称，则以 `EngineArgs` 中的设置为准。