# 与 Hugging Face 的集成

本文档介绍了 vLLM 如何与 Hugging Face 库进行集成。我们将逐步解释运行 `vllm serve` 时底层发生的具体流程。

假设我们想通过运行 `vllm serve Qwen/Qwen2-7B` 来部署热门的 Qwen 模型。

1. 首先，`model` 参数为 `Qwen/Qwen2-7B`。vLLM 会通过查找对应的配置文件 `config.json` 来判断模型是否存在。实现细节可以参考这个[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)。在这个过程中：
    - 如果 `model` 参数对应本地的某个路径，vLLM 会直接从该路径加载配置文件；
    - 如果 `model` 参数是 Hugging Face 的模型 ID（即由用户名和模型名组成），vLLM 会优先尝试从 Hugging Face 的本地缓存读取配置文件，`model` 作为模型名，`--revision` 作为版本号。关于 Hugging Face 缓存机制可参考[官方文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)；
    - 如果 `model` 参数是 Hugging Face 的模型 ID，但本地缓存中未找到，vLLM 会从 Hugging Face 模型库下载配置文件。具体实现见[此函数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)。参数包括模型名 (`model`)、版本号 (`--revision`)、以及访问模型库的令牌（环境变量 `HF_TOKEN`）。在我们的例子中，vLLM 会下载 [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) 文件。

2. 确认模型存在后，vLLM 会加载其配置文件并转换为字典。具体实现可以参考这个[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)。

3. 接下来，vLLM 会[检查](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)配置字典中的 `model_type` 字段，并据此[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)用于后续操作的配置对象。部分 `model_type` 是 vLLM 直接支持的，具体列表可见[这里](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)。如果 `model_type` 不在该列表中，vLLM 会调用 [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained) 来加载配置类，参数包括 `model`、`--revision` 和 `--trust_remote_code`。需要注意：
    - Hugging Face 也有自己的判断逻辑，会再次使用 `model_type` 字段在 transformers 库中查找对应的类名，支持的模型列表见[这里](https://github.com/huggingface/transformers/tree/main/src/transformers/models)。如果找不到，则会通过配置 JSON 文件中的 `auto_map` 字段来寻找类名，具体是 `auto_map` 下的 `AutoConfig` 字段。可以参考 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json) 作为示例。
    - `auto_map` 下的 `AutoConfig` 字段指向模型库中的模块路径，Hugging Face 会导入该模块并通过 `from_pretrained` 方法加载配置类。由于这可能会执行任意代码，仅在启用 `--trust_remote_code` 时才会执行。

4. 随后，vLLM 会对配置对象应用一些历史修补，主要涉及 RoPE（旋转位置编码）相关配置。具体实现可见[这里](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)。

5. 最后，vLLM 会根据配置对象中的 `architectures` 字段确定需要初始化的模型类。它通过[注册表](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)维护架构名与模型类的映射。如果架构名不在注册表中，说明 vLLM 暂不支持该模型架构。以 `Qwen/Qwen2-7B` 为例，`architectures` 字段为 `["Qwen2ForCausalLM"]`，对应 vLLM 代码中的 [Qwen2ForCausalLM](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364) 类。该类会根据各种配置进行初始化。

除此之外，vLLM 还依赖 Hugging Face 提供以下两个方面的支持：

1. **分词器（Tokenizer）**：vLLM 使用 Hugging Face 的分词器对输入文本进行分词。分词器通过 [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) 方法加载，参数包括模型名 (`model`) 和版本号 (`--revision`)。你也可以通过在 `vllm serve` 命令中指定 `--tokenizer` 参数来使用其他模型的分词器。相关参数还有 `--tokenizer-revision` 和 `--tokenizer-mode`，具体含义请参见 Hugging Face 的官方文档。相关逻辑可在 [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 函数中找到。获得分词器后，vLLM 会将一些高耗时属性缓存到 [get_cached_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L24)。

2. **模型权重（Model weight）**：vLLM 会从 Hugging Face 模型库下载模型权重，参数为模型名 (`model`) 和版本号 (`--revision`)。vLLM 提供了 `--load-format` 参数，用于控制从模型库下载哪些文件。默认情况下，会优先加载 safetensors 格式的权重，如果没有则回退到 PyTorch bin 格式。你也可以通过 `--load-format dummy` 跳过权重下载。
    - 推荐使用 safetensors 格式，因为它在分布式推理时加载效率高，并且可以防止执行任意代码。更多关于 safetensors 格式的信息请查阅[官方文档](https://huggingface.co/docs/safetensors/en/index)。相关逻辑可见[这里](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)。请注意：

至此，vLLM 与 Hugging Face 的集成流程就介绍完毕了。

总结来说，vLLM 会从 Hugging Face 模型库或本地目录读取 `config.json` 配置文件、分词器和模型权重。它既可以使用 vLLM 内置的配置类，也可以使用 Hugging Face transformers 提供的配置类，或者从模型库中加载自定义的配置类。