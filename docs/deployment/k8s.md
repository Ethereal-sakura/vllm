# 使用 Kubernetes

在 Kubernetes 上部署 vLLM 是一种高效且易于扩展的机器学习模型服务方案。本指南将带你通过原生 Kubernetes 部署 vLLM 的全过程。

- [CPU 部署方式](#deployment-with-cpus)
- [GPU 部署方式](#deployment-with-gpus)
- [常见问题排查](#troubleshooting)
    - [启动探针或就绪探针失败，容器日志出现 "KeyboardInterrupt: terminated"](#startup-probe-or-readiness-probe-failure-container-log-contains-keyboardinterrupt-terminated)
- [总结](#conclusion)

你也可以通过以下方式将 vLLM 部署到 Kubernetes：

- [Helm](frameworks/helm.md)
- [InftyAI/llmaz](integrations/llmaz.md)
- [KAITO](integrations/kaito.md)
- [KServe](integrations/kserve.md)
- [KubeRay](integrations/kuberay.md)
- [kubernetes-sigs/lws](frameworks/lws.md)
- [meta-llama/llama-stack](integrations/llamastack.md)
- [substratusai/kubeai](integrations/kubeai.md)
- [vllm-project/aibrix](https://github.com/vllm-project/aibrix)
- [vllm-project/production-stack](integrations/production-stack.md)

## CPU 部署方式

!!! note
    这里使用 CPU 仅用于演示和测试，实际性能无法与 GPU 相比。

首先，需要创建一个 Kubernetes PVC 和 Secret 用于下载并存储 Hugging Face 模型：

??? console "配置示例"

    ```bash
    cat <<EOF |kubectl apply -f -
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: vllm-models
    spec:
      accessModes:
        - ReadWriteOnce
      volumeMode: Filesystem
      resources:
        requests:
          storage: 50Gi
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: hf-token-secret
    type: Opaque
    data:
      token: $(HF_TOKEN)
    EOF
    ```

接下来，启动 vLLM 服务，创建 Deployment 和 Service：

??? console "配置示例"

    ```bash
    cat <<EOF |kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-server
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: vllm
      template:
        metadata:
          labels:
            app.kubernetes.io/name: vllm
        spec:
          containers:
          - name: vllm
            image: vllm/vllm-openai:latest
            command: ["/bin/sh", "-c"]
            args: [
              "vllm serve meta-llama/Llama-3.2-1B-Instruct"
            ]
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            ports:
              - containerPort: 8000
            volumeMounts:
              - name: llama-storage
                mountPath: /root/.cache/huggingface
          volumes:
          - name: llama-storage
            persistentVolumeClaim:
              claimName: vllm-models
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-server
    spec:
      selector:
        app.kubernetes.io/name: vllm
      ports:
      - protocol: TCP
        port: 8000
        targetPort: 8000
      type: ClusterIP
    EOF
    ```

你可以通过查看日志确认 vLLM 服务是否启动成功（首次下载模型可能需要几分钟时间）：

```bash
kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## GPU 部署方式

**前置要求**：请确保你已经有一个可用的 [带 GPU 的 Kubernetes 集群](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)。

1. 为 vLLM 创建 PVC、Secret 和 Deployment

      PVC 用于存储模型缓存（可选），你也可以选择 hostPath 或其它存储方式。

      <details>
      <summary>Yaml 配置</summary>

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ```

      </details>

      如果需要访问受限模型，可以创建 Secret（可选项，不使用受限模型时可跳过）。

      ```yaml
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```
  
      然后创建 vLLM 的 deployment 文件，用于运行模型服务。以下示例部署的是 `Mistral-7B-Instruct-v0.3` 模型。

      这里分别提供 NVIDIA GPU 和 AMD GPU 的示例。

      NVIDIA GPU:

      <details>
      <summary>Yaml 配置</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问宿主机的共享内存以支持张量并行推理
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            containers:
            - name: mistral-7b
              image: vllm/vllm-openai:latest
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  nvidia.com/gpu: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: cache-volume
              - name: shm
                mountPath: /dev/shm
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 5
      ```

      </details>

      AMD GPU:

      如果你使用的是 AMD ROCm GPU（如 MI300X），可以参考下面的 `deployment.yaml` 配置。

      <details>
      <summary>Yaml 配置</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            # PVC
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问宿主机的共享内存以支持张量并行推理
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "8Gi"
            hostNetwork: true
            hostIPC: true
            containers:
            - name: mistral-7b
              image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
              securityContext:
                seccompProfile:
                  type: Unconfined
                runAsGroup: 44
                capabilities:
                  add:
                  - SYS_PTRACE
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  amd.com/gpu: "1"
                requests:
                  cpu: "6"
                  memory: 6G
                  amd.com/gpu: "1"
              volumeMounts:
              - name: cache-volume
                mountPath: /root/.cache/huggingface
              - name: shm
                mountPath: /dev/shm
      ```

      </details>

      更完整的示例和步骤，可以参考 <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve>。

2. 为 vLLM 创建 Kubernetes Service

      创建 Kubernetes Service 文件，暴露 `mistral-7b` 部署服务：

      <details>
      <summary>Yaml 配置</summary>

      ```yaml
      apiVersion: v1
      kind: Service
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        ports:
        - name: http-mistral-7b
          port: 80
          protocol: TCP
          targetPort: 8000
        # selector 需与 deployment 标签一致，同时有助于前缀缓存功能
        selector:
          app: mistral-7b
        sessionAffinity: None
        type: ClusterIP
      ```

      </details>

3. 部署并测试

      使用 `kubectl apply -f <filename>` 应用 deployment 和 service 配置：

      ```bash
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      可以通过下面的 `curl` 命令测试服务是否正常：

      ```bash
      curl http://mistral-7b.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "mistralai/Mistral-7B-Instruct-v0.3",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      如果服务正常部署，你会收到来自 vLLM 模型的响应。

## 常见问题排查

### 启动探针或就绪探针失败，容器日志出现 "KeyboardInterrupt: terminated"

如果启动探针（startup probe）或就绪探针（readiness probe）的 failureThreshold 设置过低，导致服务启动时间不够，Kubernetes 调度器就会终止容器。通常会有如下现象：

1. 容器日志中包含 "KeyboardInterrupt: terminated"
2. 执行 `kubectl get events` 时出现 `Container $NAME failed startup probe, will be restarted` 信息

解决方法：提高 failureThreshold，给予模型服务更多的启动时间。你可以先将探针配置去掉，观察模型服务完全启动并可用需要多长时间，再据此调整 failureThreshold 的数值。

## 总结

通过 Kubernetes 部署 vLLM，可以高效利用 GPU 资源，实现大模型的弹性扩展和统一管理。按照上述步骤，你可以在自己的 Kubernetes 集群中快速搭建并测试 vLLM 服务。如有遇到问题或有更好的建议，欢迎为文档贡献改进意见。