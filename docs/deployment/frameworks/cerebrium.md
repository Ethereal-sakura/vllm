# Cerebrium

<p align="center">
    <img src="https://i.ibb.co/hHcScTT/Screenshot-2024-06-13-at-10-14-54.png" alt="vLLM_plus_cerebrium"/>
</p>

你可以通过 [Cerebrium](https://www.cerebrium.ai/) 在云端 GPU 服务器上运行 vLLM。Cerebrium 是一个无服务器（serverless）AI 基础设施平台，让企业更轻松地构建并部署基于 AI 的应用。

要安装 Cerebrium 客户端，请执行：

```bash
pip install cerebrium
cerebrium login
```

接下来，创建你的 Cerebrium 项目，运行：

```bash
cerebrium init vllm-project
```

然后，为了安装所需依赖，在你的 cerebrium.toml 文件中添加如下内容：

```toml
[cerebrium.deployment]
docker_base_image_url = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

[cerebrium.dependencies.pip]
vllm = "latest"
```

之后，我们来添加推理代码，用于调用你选择的大语言模型（LLM，这里以 `mistralai/Mistral-7B-Instruct-v0.1` 为例）。将以下代码添加到你的 `main.py` 文件中：

??? code

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

    def run(prompts: list[str], temperature: float = 0.8, top_p: float = 0.95):

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
        outputs = llm.generate(prompts, sampling_params)

        # 打印输出结果。
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append({"prompt": prompt, "generated_text": generated_text})

        return {"results": results}
    ```

之后，运行以下命令将其部署到云端：

```bash
cerebrium deploy
```

如果部署成功，你会收到一个 CURL 命令，可以用来调用你的模型接口进行推理。注意，URL 结尾要加上你要调用的函数名（这里是 `/run`）

??? console "命令示例"

    ```bash
    curl -X POST https://api.cortex.cerebrium.ai/v4/p-xxxxxx/vllm/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: <JWT TOKEN>' \
    --data '{
    "prompts": [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is"
    ]
    }'
    ```

你将会收到类似如下的响应：

??? console "响应示例"

    ```json
    {
        "run_id": "52911756-3066-9ae8-bcc9-d9129d1bd262",
        "result": {
            "result": [
                {
                    "prompt": "Hello, my name is",
                    "generated_text": " Sarah, and I'm a teacher. I teach elementary school students. One of"
                },
                {
                    "prompt": "The president of the United States is",
                    "generated_text": " elected every four years. This is a democratic system.\n\n5. What"
                },
                {
                    "prompt": "The capital of France is",
                    "generated_text": " Paris.\n"
                },
                {
                    "prompt": "The future of AI is",
                    "generated_text": " bright, but it's important to approach it with a balanced and nuanced perspective."
                }
            ]
        },
        "run_time_ms": 152.53663063049316
    }
    ```

现在你已经拥有了一个自动扩缩容的推理接口，按实际用量计费！