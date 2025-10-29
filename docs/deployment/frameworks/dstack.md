# dstack

<p align="center">
    <img src="https://i.ibb.co/71kx6hW/vllm-dstack.png" alt="vLLM_plus_dstack"/>
</p>

你可以通过 [dstack](https://dstack.ai/) —— 一个开源框架，在任意云服务上运行 LLM（大语言模型，Large Language Models），在云端 GPU 机器上运行 vLLM。本教程假定你已经在云环境中完成了凭证、网关和 GPU 配额的配置。

安装 dstack 客户端，请运行：

```bash
pip install dstack[all]
dstack server
```

接下来，配置你的 dstack 项目，运行：

```bash
mkdir -p vllm-dstack
cd vllm-dstack
dstack init
```

然后，为你选择的 LLM（本例使用 `NousResearch/Llama-2-7b-chat-hf`），在 dstack 的 `Service` 中创建如下 `serve.dstack.yml` 文件，以便自动部署虚拟机实例：

??? code "配置"

    ```yaml
    type: service

    python: "3.11"
    env:
        - MODEL=NousResearch/Llama-2-7b-chat-hf
    port: 8000
    resources:
        gpu: 24GB
    commands:
        - pip install vllm
        - vllm serve $MODEL --port 8000
    model:
        format: openai
        type: chat
        name: NousResearch/Llama-2-7b-chat-hf
    ```

之后，使用以下命令行来启动部署：

??? console "命令"

    ```console
    $ dstack run . -f serve.dstack.yml

    ⠸ 获取运行计划...
    配置        serve.dstack.yml
    项目        deep-diver-main
    用户        deep-diver
    最低资源    2..xCPU, 8GB.., 1xGPU (24GB)
    最高价格    -
    最长时长    -
    Spot 策略   auto
    重试策略    no

    #  BACKEND  REGION       INSTANCE       RESOURCES                               SPOT  PRICE
    1  gcp   us-central1  g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    2  gcp   us-east1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    3  gcp   us-west1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
        ...
    已显示 3 条，共 193 个资源报价，最高 $5.876

    是否继续？ [y/n]: y
    ⠙ 正在提交运行...
    ⠏ 正在启动 spicy-treefrog-1（拉取镜像中）
    spicy-treefrog-1 资源已部署完成（运行中）
    服务已发布于 ...
    ```

部署完成后，你可以使用 OpenAI SDK 与模型进行交互：

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="https://gateway.<gateway domain>",
        api_key="<YOUR-DSTACK-SERVER-ACCESS-TOKEN>",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Llama-2-7b-chat-hf",
        messages=[
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming.",
            }
        ],
    )

    print(completion.choices[0].message.content)
    ```

!!! note
    dstack 会自动使用自己的 token 在网关上完成认证。如果不想配置网关，也可以选择部署 dstack 的 `Task`，适合开发测试使用。更多关于如何用 dstack 部署 vLLM 的详细动手教程，请查看 [这个仓库](https://github.com/dstackai/dstack-examples/tree/main/deployment/vllm)