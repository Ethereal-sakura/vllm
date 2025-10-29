# SkyPilot

<p align="center">
  <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
</p>

通过 [SkyPilot](https://github.com/skypilot-org/skypilot) 这个开源框架，vLLM 可以在任意云平台或 Kubernetes 上**一键部署并扩展为多副本服务**。你可以在 [SkyPilot AI gallery](https://skypilot.readthedocs.io/en/latest/gallery/index.html) 找到更多关于不同开源模型（如 Llama-3、Mixtral 等）的示例。

## 前置条件

- 前往 [HuggingFace 模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 并申请访问 `meta-llama/Meta-Llama-3-8B-Instruct` 模型。
- 确认你已经安装好了 SkyPilot（[安装说明](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)）。
- 运行 `sky check` 检查云平台或 Kubernetes 是否已启用。

```bash
pip install skypilot-nightly
sky check
```

## 单机部署

查看 vLLM 的 SkyPilot 服务部署 YAML 配置，[serving.yaml](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml)。

??? code "Yaml"

    ```yaml
    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可以选择更便宜的加速卡。
      use_spot: True
      disk_size: 512  # 确保磁盘容量足够存放模型权重文件。
      disk_tier: best
      ports: 8081  # 对外开放此端口以供访问。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为你自己的 huggingface token，或通过 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 以启用网页界面。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo '正在启动 vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log &

      echo '等待 vllm api server 启动完成...'
      while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done

      echo '正在启动 gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/online_serving/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://localhost:8081/v1 \
        --stop-token-ids 128009,128001
    ```

使用上述任意一种 GPU（L4、A10g 等）启动 Llama-3 8B 模型服务：

```bash
HF_TOKEN="your-huggingface-token" sky launch serving.yaml --env HF_TOKEN
```

查看命令输出。你会看到一个 gradio 的分享链接（如最后一行）。在浏览器中打开这个链接，即可用 LLaMA 进行文本补全。

```console
(task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live
```

**可选**：如需部署 70B 大模型并使用更多 GPU，可按如下方式：

```bash
HF_TOKEN="your-huggingface-token" \
  sky launch serving.yaml \
  --gpus A100:8 \
  --env HF_TOKEN \
  --env MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
```

## 多副本扩展

SkyPilot 支持自动扩缩、多副本负载均衡和容错。只需在 YAML 文件中增加 `services` 配置即可。

??? code "Yaml"

    ```yaml
    service:
      replicas: 2
      # 实际的健康探测请求。
      readiness_probe:
        path: /v1/chat/completions
        post_data:
        model: $MODEL_NAME
        messages:
          - role: user
            content: Hello! What is your name?
      max_completion_tokens: 1
    ```

??? code "Yaml"

    ```yaml
    service:
      replicas: 2
      # 实际的健康探测请求。
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可以选择更便宜的加速卡。
      use_spot: True
      disk_size: 512  # 确保磁盘容量足够存放模型权重文件。
      disk_tier: best
      ports: 8081  # 对外开放此端口以供访问。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为你自己的 huggingface token，或通过 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 以启用网页界面。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo '正在启动 vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

启动多副本 Llama-3 8B 模型服务：

```bash
HF_TOKEN="your-huggingface-token" \
  sky serve up -n vllm serving.yaml \
  --env HF_TOKEN
```

等待服务就绪：

```bash
watch -n10 sky serve status vllm
```

示例输出：

```console
Services
NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
vllm  1        35s     READY   2/2       xx.yy.zz.100:30001

Service Replicas
SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
vllm          1   1        xx.yy.zz.121  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
vllm          2   1        xx.yy.zz.245  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
```

当服务状态为 READY 后，你可以通过统一的 endpoint 访问服务：

??? console "命令示例"

    ```bash
    ENDPOINT=$(sky serve status --endpoint 8081 vllm)
    curl -L http://$ENDPOINT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Who are you?"
        }
        ],
        "stop_token_ids": [128009,  128001]
      }'
    ```

如需启用自动扩缩容，可以将 `replicas` 替换为以下配置：

```yaml
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 2
```

当单个副本每秒请求数超过 2 时，服务会自动扩容。

??? code "Yaml"

    ```yaml
    service:
      replica_policy:
        min_replicas: 2
        max_replicas: 4
        target_qps_per_replica: 2
      # 实际的健康探测请求。
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可以选择更便宜的加速卡。
      use_spot: True
      disk_size: 512  # 确保磁盘容量足够存放模型权重文件。
      disk_tier: best
      ports: 8081  # 对外开放此端口以供访问。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为你自己的 huggingface token，或通过 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 以启用网页界面。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo '正在启动 vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

如需更新服务配置：

```bash
HF_TOKEN="your-huggingface-token" sky serve update vllm serving.yaml --env HF_TOKEN
```

关闭服务：

```bash
sky serve down vllm
```

### **可选**：通过 GUI 访问服务

你也可以将 Llama-3 服务与独立的网页前端（GUI）结合，这样用户的请求会自动被负载均衡到各个副本。

??? code "Yaml"

    ```yaml
    envs:
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      ENDPOINT: x.x.x.x:3031 # 运行 vllm 的 API 服务地址

    resources:
      cpus: 2

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      # 安装 Gradio 以启用网页界面。
      pip install gradio openai

    run: |
      conda activate vllm
      export PATH=$PATH:/sbin

      echo '正在启动 gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/online_serving/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://$ENDPOINT/v1 \
        --stop-token-ids 128009,128001 | tee ~/gradio.log
    ```

1. 启动聊天网页 UI：

    ```bash
    sky launch \
      -c gui ./gui.yaml \
      --env ENDPOINT=$(sky serve status --endpoint vllm)
    ```

2. 然后，你可以在输出的 gradio 链接访问 GUI 页面：

    ```console
    | INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live
    ```