# 快速入门

本指南将帮助你快速开始使用 vLLM，支持以下场景：

- [离线批量推理](#offline-batched-inference)
- [基于 OpenAI 协议的在线服务](#openai-compatible-server)

## 前置条件

- 操作系统：Linux
- Python：3.10 -- 3.13

## 安装指南

=== "NVIDIA CUDA"

    如果你使用的是 NVIDIA GPU，可以直接通过 [pip](https://pypi.org/project/vllm/) 安装 vLLM。

    推荐使用 [uv](https://docs.astral.sh/uv/)，这是一款高速的 Python 环境管理工具，可以帮助你创建和管理 Python 环境。请参考 [官方文档](https://docs.astral.sh/uv/#getting-started) 完成 `uv` 的安装。安装完成后，可以通过以下命令新建环境并安装 vLLM：

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

    `uv` 可以通过 `--torch-backend=auto`（或设置环境变量 `UV_TORCH_BACKEND=auto`）自动检测你的 CUDA 驱动版本，从而自动选择合适的 PyTorch 后端。你也可以指定具体后端（如 `cu126`），只需设置 `--torch-backend=cu126`（或 `UV_TORCH_BACKEND=cu126`）即可。

    另外一个便捷方式是使用 `uv run` 搭配 `--with [dependency]`，这样可以直接运行如 `vllm serve` 命令，无需创建持久化环境：

    ```bash
    uv run --with vllm vllm --help
    ```

    你也可以通过 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 创建和管理 Python 环境。如果希望在 conda 环境内管理 `uv`，可以用 `pip` 安装：

    ```bash
    conda create -n myenv python=3.12 -y
    conda activate myenv
    pip install --upgrade uv
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"

    建议直接使用 Docker Hub 上的预构建镜像。稳定版镜像为 [rocm/vllm:latest](https://hub.docker.com/r/rocm/vllm)，开发版镜像为 [rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev)。
    
    在以下 `docker run` 命令中，`-v` 参数用于将本地目录挂载到容器。请将 `<path/to/your/models>` 替换为主机上存放模型的路径，模型会在容器内映射为 `/app/models`。
    
    ???+ console "命令示例"
        ```bash
        docker pull rocm/vllm-dev:nightly # 拉取最新镜像
        docker run -it --rm \
        --network=host \
        --group-add=video \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v <path/to/your/models>:/app/models \
        -e HF_HOME="/app/models" \
        rocm/vllm-dev:nightly
        ```

!!! note
    更多详细信息及非 CUDA 平台的安装方法，请参考[这里](installation/README.md)。

## 离线批量推理

安装 vLLM 后，你可以为一组输入 prompt 批量生成文本（即离线推理）。可参考示例脚本：[examples/offline_inference/basic/basic.py](../../examples/offline_inference/basic/basic.py)

示例代码的第一行导入了 [LLM][vllm.LLM] 和 [SamplingParams][vllm.SamplingParams]：

- [LLM][vllm.LLM]：用于运行 vLLM 推理引擎的主类
- [SamplingParams][vllm.SamplingParams]：用于配置采样过程的参数

```python
from vllm import LLM, SamplingParams
```

接下来定义了输入 prompt 列表和文本生成的采样参数。采样温度（sampling temperature）设置为 `0.8`，[核采样概率（nucleus sampling probability）](https://en.wikipedia.org/wiki/Top-p_sampling)设为 `0.95`。关于采样参数的更多信息，请参考[这里](../api/README.md#inference-parameters)。

!!! important
    默认情况下，vLLM 会优先使用 Hugging Face 模型仓库中的 `generation_config.json` 来设置推荐的采样参数。如果未指定 [SamplingParams][vllm.SamplingParams]，通常会获得更优的默认推理效果。

    如果你希望使用 vLLM 默认采样参数，在创建 [LLM][vllm.LLM] 实例时请设置 `generation_config="vllm"`。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

[LLM][vllm.LLM] 类会初始化 vLLM 引擎，并加载 [OPT-125M 模型](https://arxiv.org/abs/2205.01068)用于离线推理。支持的模型列表见[这里](../models/supported_models.md)。

```python
llm = LLM(model="facebook/opt-125m")
```

!!! note
    vLLM 默认从 [Hugging Face](https://huggingface.co/) 下载模型。如果你希望从 [ModelScope](https://www.modelscope.cn) 下载模型，可在初始化前设置环境变量：

    ```shell
    export VLLM_USE_MODELSCOPE=True
    ```

现在进入生成环节！使用 `llm.generate` 进行推理，输入 prompt 会被加入 vLLM 的队列并高效生成输出。输出结果是一个 `RequestOutput` 对象列表，包含所有生成的 token。

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! note
    `llm.generate` 方法不会自动将模型的聊天模板（chat template）应用到输入 prompt。如果你使用的是 Instruct 或 Chat 模型，建议手动应用对应的聊天模板，确保推理效果符合预期。或者你可以使用 `llm.chat` 方法，传入和 OpenAI `client.chat.completions` 相同格式的消息列表：

    ??? code
    
        ```python
        # 使用 tokenizer 应用聊天模板
        from transformers import AutoTokenizer
    
        tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 生成输出
        outputs = llm.generate(texts, sampling_params)
        
        # 打印结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
        # 使用 chat 接口
        outputs = llm.chat(messages_list, sampling_params)
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        ```

## OpenAI 协议兼容服务

vLLM 可作为支持 OpenAI API 协议的服务部署。这样你可以直接用 vLLM 替换任何基于 OpenAI API 的应用。服务默认监听 `http://localhost:8000`，可以通过 `--host` 和 `--port` 参数自定义地址。服务当前一次只支持加载一个模型，并实现了如 [模型列表](https://platform.openai.com/docs/api-reference/models/list)、[创建聊天补全](https://platform.openai.com/docs/api-reference/chat/completions/create)、[创建文本补全](https://platform.openai.com/docs/api-reference/completions/create) 等接口。

以下命令可启动 vLLM 服务，加载 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

!!! note
    服务将默认使用 tokenizer 中预设的聊天模板。
    如何自定义聊天模板，请参考[这里](../serving/openai_compatible_server.md#chat-template)。
!!! important
    默认情况下，服务会自动应用 huggingface 仓库中的 `generation_config.json`，即采样参数的默认值可能会被模型作者推荐的参数覆盖。

    如需禁用此行为，请在启动服务时增加 `--generation-config vllm` 参数。

服务的调用格式与 OpenAI API 完全一致。例如查询模型列表：

```bash
curl http://localhost:8000/v1/models
```

你可以通过参数 `--api-key` 或环境变量 `VLLM_API_KEY` 启用 API 密钥校验。支持同时配置多个密钥，服务会接受其中任意一个，这对于密钥轮换非常方便。

### 使用 vLLM 对接 OpenAI Completions API

服务启动后，可以直接用 prompt 查询模型：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

由于服务完全兼容 OpenAI API，你可以无缝替换任何用 OpenAI API 的应用。例如，也可以用 `openai` 官方 Python 包进行调用：

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和地址，指向 vLLM 服务
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="San Francisco is a",
    )
    print("Completion result:", completion)
    ```

更详细的客户端示例可参考：[examples/offline_inference/basic/basic.py](../../examples/offline_inference/basic/basic.py)

### 使用 vLLM 对接 OpenAI Chat Completions API

vLLM 也支持 OpenAI 聊天补全 API。聊天接口更适合动态、交互式的对话，可保存聊天历史，非常适合需要上下文或详细解释的场景。

你可以通过 [创建聊天补全](https://platform.openai.com/docs/api-reference/chat/completions/create) 接口与模型交互：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

同样可以用 `openai` Python 包调用：

??? code

    ```python
    from openai import OpenAI
    # 设置 OpenAI API key 和地址，指向 vLLM 服务
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    print("Chat response:", chat_response)
    ```

## 关于 Attention 后端

目前，vLLM 支持多种高效 Attention（注意力机制）后端，兼容不同平台和加速器架构。系统会自动为你的环境和模型选择最优的后端。

如需手动指定 Attention 后端，可设置环境变量 `VLLM_ATTENTION_BACKEND`，支持选项如下：

- NVIDIA CUDA 下可选：`FLASH_ATTN`、`FLASHINFER` 或 `XFORMERS`
- AMD ROCm 下可选：`TRITON_ATTN`、`ROCM_ATTN`、`ROCM_AITER_FA` 或 `ROCM_AITER_UNIFIED_ATTN`

对于 AMD ROCm，还可以通过以下变量更细致地控制 Attention 实现：

- Triton 统一 Attention：`VLLM_ROCM_USE_AITER=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 VLLM_ROCM_USE_AITER_MHA=0`
- AITER 统一 Attention：`VLLM_ROCM_USE_AITER=1 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 VLLM_ROCM_USE_AITER_MHA=0`
- Triton Prefill-Decode Attention：`VLLM_ROCM_USE_AITER=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0`
- AITER 多头 Attention：`VLLM_ROCM_USE_AITER=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 VLLM_ROCM_USE_AITER_MHA=1`

!!! warning
    目前没有预编译包含 Flash Infer 的 vllm 安装包，请你先在环境中手动安装。安装方法可参考 [Flash Infer 官方文档](https://docs.flashinfer.ai/) 或 [docker/Dockerfile](../../docker/Dockerfile)。