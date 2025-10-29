# Hugging Face Inference Endpoints

## 概览

兼容 vLLM 的模型可以通过 Hugging Face Inference Endpoints 部署，无论是从 [Hugging Face Hub](https://huggingface.co) 启动，还是直接在 [Inference Endpoints](https://endpoints.huggingface.co/) 页面操作都可以。这样，你可以在 Hugging Face 提供的全托管环境中部署模型，享受 GPU 加速、自动扩容和实时监控，无需自己管理底层基础设施。

如果你需要更深入的 vLLM 集成与部署方式信息，请参考[高级部署细节](#advanced-deployment-details)。

## 部署方式

- [**方式一：通过模型目录一键部署。**](#method-1-deploy-from-the-catalog) 直接在 Hugging Face Hub 的目录中选择已经优化好的模型进行一键部署。
- [**方式二：引导式部署（Transformers 模型）。**](#method-2-guided-deployment-transformers-models) 对于 Hub 上带有 `transformers` 标签的模型，可以在页面上通过 **Deploy** 按钮快速部署。
- [**方式三：手动部署（高级自定义模型）。**](#method-3-manual-deployment-advanced-models) 针对带有 `transformers` 标签但使用自定义代码，或不兼容标准 `transformers` 但受 vLLM 支持的模型，需要手动配置部署。

### 方式一：通过模型目录一键部署

这是在 Hugging Face Inference Endpoints 上体验 vLLM 的最简单方法。你可以在 [Inference Endpoints](https://endpoints.huggingface.co/catalog) 的模型目录中浏览已经验证和优化过的部署配置，最大化发挥模型性能。

1. 打开 [Endpoints Catalog](https://endpoints.huggingface.co/catalog)，在 **Inference Server** 选项中选择 `vLLM`。这样会展示所有已优化并预配置的可用模型。

    ![Endpoints Catalog](../../assets/deployment/hf-inference-endpoints-catalog.png)

2. 选中你需要的模型，点击 **Create Endpoint**。

    ![Create Endpoint](../../assets/deployment/hf-inference-endpoints-create-endpoint.png)

3. 部署完成后，即可使用该 endpoint。将控制台提供的 URL 填写到 `DEPLOYMENT_URL`，注意按要求结尾加上 `/v1`。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="HuggingFaceTB/SmolLM3-3B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "用简单的话解释一下重力是什么。",
                    }
                ],
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    目录中的模型都已针对 vLLM 进行了优化，包括 GPU 设置和推理引擎配置。你可以在 Inference Endpoints 控制台中监控 endpoint 状态，并随时更新 **容器或其配置**。

### 方式二：引导式部署（Transformers 模型）

本方法适用于在模型元数据中带有 [`transformers` 库标签](https://huggingface.co/models?library=transformers) 的模型。你可以直接在 Hugging Face Hub 页面，无需手动配置快速部署。

1. 打开 [Hugging Face Hub](https://huggingface.co/models) 并进入你感兴趣的模型页面。  
   以 [`ibm-granite/granite-docling-258M`](https://huggingface.co/ibm-granite/granite-docling-258M) 为例。你可以在该模型 [README](https://huggingface.co/ibm-granite/granite-docling-258M/blob/main/README.md) 的前言部分确认 `library: transformers` 标签。

2. 找到 **Deploy** 按钮。对于带有 `transformers` 标签的模型，在模型卡右上角会显示该按钮。

    ![Locate deploy button](../../assets/deployment/hf-inference-endpoints-locate-deploy-button.png)

3. 点击 **Deploy** 按钮，选择 **HF Inference Endpoints**，系统会跳转到 Inference Endpoints 配置界面。

    ![Click deploy button](../../assets/deployment/hf-inference-endpoints-click-deploy-button.png)

4. 选择硬件（本例选择 AWS>GPU>T4）及容器配置。记得将容器类型设为 `vLLM`，然后点击 **Create Endpoint** 完成部署。

    ![Select Hardware](../../assets/deployment/hf-inference-endpoints-select-hardware.png)

5. 使用已部署的 endpoint。将控制台提供的 URL 填入 `DEPLOYMENT_URL`（记得加上 `/v1`），即可通过程序或 SDK 调用。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="ibm-granite/granite-docling-258M",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请将这页内容转换为 docling。",
                    },
                ]
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    此方法会采用默认推荐配置。如果有特殊需求，建议根据实际情况调整参数。

### 方式三：手动部署（高级自定义模型）

部分模型需要手动部署，通常原因包括：

- 使用了带 `transformers` 标签的自定义代码
- 不兼容标准 `transformers`，但已被 `vLLM` 支持

这些模型无法直接通过模型卡的 **Deploy** 按钮部署。

本例中，我们将以 [`rednote-hilab/dots.ocr`](https://huggingface.co/rednote-hilab/dots.ocr) 作为演示对象，这是一款已集成 vLLM 的 OCR 模型（详细见 vLLM [PR](https://github.com/vllm-project/vllm/pull/24645)）。

1. 新建部署。在 [Inference Endpoints](https://endpoints.huggingface.co/) 页面点击 `New`。

    ![New Endpoint](../../assets/deployment/hf-inference-endpoints-new-endpoint.png)

2. 在 Hub 中搜索模型。在弹出的对话框切换到 **Hub**，输入你想要部署的模型名称进行搜索。

    ![Select model](../../assets/deployment/hf-inference-endpoints-select-model.png)

3. 选择基础设施。在配置页面选择云服务商和硬件类型。  
   本示例选择 AWS 的 L4 GPU，实际可根据你的需求调整。

    ![Choose Infra](../../assets/deployment/hf-inference-endpoints-choose-infra.png)

4. 配置容器。下拉到 **Container Configuration**，将容器类型选择为 `vLLM`。

    ![Configure Container](../../assets/deployment/hf-inference-endpoints-configure-container.png)

5. 创建 endpoint。点击 **Create Endpoint** 开始部署模型。

    部署完成后，可以通过 OpenAI Completion API、cURL 或其他 SDK 调用 endpoint。需要时请在 URL 末尾添加 `/v1`。

!!! note
    你可以在 Inference Endpoints 控制台修改 **容器设置**（Container URI、Container Arguments），并通过 **Update Endpoint** 应用变更。这会用新的容器配置重新部署 endpoint。如果需要更换模型本身，则须新建 endpoint 或重新部署。在本例中，可能需要将 Container URI 更新为 nightly 镜像（`vllm/vllm-openai:nightly`），并在容器参数中添加 `--trust-remote-code` 标志。

## 高级部署细节

借助 [transformers 后端集成](https://blog.vllm.ai/2025/04/11/transformers-backend.html)，vLLM 现已实现对所有兼容 `transformers` 的模型开箱即用（Day 0 支持）。你可以直接部署这些模型，充分利用 vLLM 优化的推理能力，无需改动后端代码。

Hugging Face Inference Endpoints 为 vLLM 提供了全托管的模型服务环境。你无需配置服务器、安装依赖或管理集群，即可部署模型。Endpoints 支持在多个云平台（AWS、Azure、GCP）上部署，无需额外注册账号。

平台与 Hugging Face Hub 紧密集成，允许你随时部署任何支持 vLLM 或 `transformers` 的模型，监控用量并直接更新推理引擎。vLLM 引擎已预先配置好，既能高效推理，也便于在不同模型和引擎之间切换，无需修改业务代码。这种方式大大简化了线上部署流程：endpoint 数分钟即可就绪，并内置监控及日志功能，让你专注于模型服务本身，无需关心底层运维。

## 后续建议

- 浏览 [Inference Endpoints](https://endpoints.huggingface.co/catalog) 模型目录
- 阅读 Inference Endpoints 的[官方文档](https://huggingface.co/docs/inference-endpoints/en/index)
- 了解 [Inference Endpoints 引擎](https://huggingface.co/docs/inference-endpoints/en/engines/vllm)
- 深入学习 [transformers 后端集成](https://blog.vllm.ai/2025/04/11/transformers-backend.html)
