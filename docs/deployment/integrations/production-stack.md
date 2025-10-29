# 生产环境部署栈

在 Kubernetes 上部署 vLLM，是一种高效且可扩展的机器学习模型服务方式。本指南将带你一步步使用 [vLLM production stack](https://github.com/vllm-project/production-stack) 完成部署。这个部署栈源自伯克利和芝加哥大学的合作，是 [vLLM 项目](https://github.com/vllm-project) 官方发布、专为大语言模型（LLM）生产环境优化的代码库，具备以下特点：

* **与上游 vLLM 完全兼容** —— 无需修改上游 vLLM 代码，直接封装使用。
* **简单易用** —— 通过 Helm chart 实现一键部署，并可结合 Grafana 仪表盘进行监控。
* **高性能** —— 针对 LLM 场景优化，支持多模型、模型感知和前缀感知路由、快速启动、结合 [LMCache](https://github.com/LMCache/LMCache) 实现 KV 缓存卸载等特性。

如果你是 Kubernetes 新手也不用担心：在 vLLM production stack 的 [仓库](https://github.com/vllm-project/production-stack) 中，我们提供了详细的[安装教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)和[4 分钟上手视频](https://www.youtube.com/watch?v=EsTJbQtzj0g)，帮你快速搭建环境！

## 前置条件

请确保你已经拥有带有 GPU 的 Kubernetes 环境（你可以按照[本教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)在物理机上搭建 Kubernetes 环境）。

## 使用 vLLM production stack 部署

标准的 vLLM production stack 通过 Helm chart 进行安装。你可以在 GPU 服务器上运行这个 [bash 脚本](https://github.com/vllm-project/production-stack/blob/main/utils/install-helm.sh) 来安装 Helm。

在你的桌面执行以下命令，即可安装 vLLM production stack：

```bash
sudo helm repo add vllm https://vllm-project.github.io/production-stack
sudo helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```

这会基于 vLLM-production-stack 启动一个名为 `vllm` 的部署，默认运行一个小型 LLM（Facebook opt-125M 模型）。

### 验证安装

你可以通过以下命令监控部署状态：

```bash
sudo kubectl get pods
```

此时你会看到 `vllm` 部署下的 pod 状态变为 `Running`。

```text
NAME                                           READY   STATUS    RESTARTS   AGE
vllm-deployment-router-859d8fb668-2x2b7        1/1     Running   0          2m38s
vllm-opt125m-deployment-vllm-84dfc9bd7-vb9bs   1/1     Running   0          2m38s
```

!!! note
    容器启动并下载 Docker 镜像及 LLM 权重文件可能需等待一段时间。

### 发送请求测试服务

将 `vllm-router-service` 的端口转发到本地：

```bash
sudo kubectl port-forward svc/vllm-router-service 30080:80
```

然后你可以通过 OpenAI 兼容 API 查询当前可用模型：

```bash
curl -o- http://localhost:30080/v1/models
```

??? console "输出示例"

    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "facebook/opt-125m",
          "object": "model",
          "created": 1737428424,
          "owned_by": "vllm",
          "root": null
        }
      ]
    }
    ```

如果你想直接发送对话请求，可以调用 OpenAI `/completion` 接口：

```bash
curl -X POST http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```

??? console "输出示例"

    ```json
    {
      "id": "completion-id",
      "object": "text_completion",
      "created": 1737428424,
      "model": "facebook/opt-125m",
      "choices": [
        {
          "text": " there was a brave knight who...",
          "index": 0,
          "finish_reason": "length"
        }
      ]
    }
    ```

### 卸载部署

如需移除部署，执行以下命令：

```bash
sudo helm uninstall vllm
```

---

### （进阶）自定义 vLLM production stack 配置

vLLM production stack 的核心配置通过 YAML 文件进行管理。以下是前文安装过程所用的配置示例：

??? code "Yaml"

    ```yaml
    servingEngineSpec:
      runtimeClassName: ""
      modelSpec:
      - name: "opt125m"
        repository: "vllm/vllm-openai"
        tag: "latest"
        modelURL: "facebook/opt-125m"

        replicaCount: 1

        requestCPU: 6
        requestMemory: "16Gi"
        requestGPU: 1

        pvcStorage: "10Gi"
    ```

在这个 YAML 配置中：

* **`modelSpec`** 包含：
    * `name`：你自定义的模型别名。
    * `repository`：vLLM 的 Docker 镜像仓库。
    * `tag`：Docker 镜像标签。
    * `modelURL`：你想运行的 LLM 模型。
* **`replicaCount`**：副本数量。
* **`requestCPU` 和 `requestMemory`**：指定 pod 需要的 CPU 和内存资源。
* **`requestGPU`**：需要的 GPU 数量。
* **`pvcStorage`**：为模型分配的持久化存储空间。

!!! note
    如果你希望部署两个 pod，请参考这个 [YAML 文件](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml)。

!!! tip
    vLLM production stack 还支持更多高级特性（如 CPU 卸载、多种路由算法等）。欢迎查阅更多[示例与教程](https://github.com/vllm-project/production-stack/tree/main/tutorials)以及我们的[仓库](https://github.com/vllm-project/production-stack)了解详情！