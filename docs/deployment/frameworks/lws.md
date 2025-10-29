# LWS

LeaderWorkerSet（LWS）是一个 Kubernetes API，专门用于解决 AI/ML 推理场景下常见的部署模式。
其中一个主要应用场景就是跨主机、多节点的分布式推理任务。

vLLM 可以通过在 Kubernetes 上结合 [LWS](https://github.com/kubernetes-sigs/lws) 实现分布式模型服务。

## 前置条件

* 需要至少两台 Kubernetes 节点，每台配备 8 块 GPU。
* 按照 [官方文档](https://lws.sigs.k8s.io/docs/installation/) 的说明安装 LWS。

## 部署与服务

部署下面的 yaml 文件 `lws.yaml`

??? code "Yaml"

    ```yaml
    apiVersion: leaderworkerset.x-k8s.io/v1
    kind: LeaderWorkerSet
    metadata:
      name: vllm
    spec:
      replicas: 1
      leaderWorkerTemplate:
        size: 2
        restartPolicy: RecreateGroupOnPodRestart
        leaderTemplate:
          metadata:
            labels:
              role: leader
          spec:
            containers:
              - name: vllm-leader
                image: docker.io/vllm/vllm-openai:latest
                env:
                  - name: HF_TOKEN
                    value: <your-hf-token>
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=$(LWS_GROUP_SIZE); 
                    vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --port 8080 --tensor-parallel-size 8 --pipeline_parallel_size 2"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                ports:
                  - containerPort: 8080
                readinessProbe:
                  tcpSocket:
                    port: 8080
                  initialDelaySeconds: 15
                  periodSeconds: 10
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
        workerTemplate:
          spec:
            containers:
              - name: vllm-worker
                image: docker.io/vllm/vllm-openai:latest
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker --ray_address=$(LWS_LEADER_ADDRESS)"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                env:
                  - name: HF_TOKEN
                    value: <your-hf-token>
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm   
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-leader
    spec:
      ports:
        - name: http
          port: 8080
          protocol: TCP
          targetPort: 8080
      selector:
        leaderworkerset.sigs.k8s.io/name: vllm
        role: leader
      type: ClusterIP
    ```

```bash
kubectl apply -f lws.yaml
```

查看 Pod 的运行状态：

```bash
kubectl get pods
```

你会看到类似如下的输出：

```bash
NAME       READY   STATUS    RESTARTS   AGE
vllm-0     1/1     Running   0          2s
vllm-0-1   1/1     Running   0          2s
```

验证分布式张量并行推理是否正常：

```bash
kubectl logs vllm-0 |grep -i "Loading model weights took" 
```

输出内容大致如下：

```text
INFO 05-08 03:20:24 model_runner.py:173] Loading model weights took 0.1189 GB
(RayWorkerWrapper pid=169, ip=10.20.0.197) INFO 05-08 03:20:28 model_runner.py:173] Loading model weights took 0.1189 GB
```

## 访问 ClusterIP 服务

```bash
# 本地监听 8080 端口，并转发到集群中被服务选中的 Pod 的 8080 端口
kubectl port-forward svc/vllm-leader 8080:8080
```

你会看到类似如下的输出：

```text
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

## 调用模型服务

打开另一个终端，发送请求：

```text
curl http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
}'
```

输出内容大致如下

??? console "输出结果"

    ```text
    {
      "id": "cmpl-1bb34faba88b43f9862cfbfb2200949d",
      "object": "text_completion",
      "created": 1715138766,
      "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
      "choices": [
        {
          "index": 0,
          "text": " top destination for foodies, with",
          "logprobs": null,
          "finish_reason": "length",
          "stop_reason": null
        }
      ],
      "usage": {
        "prompt_tokens": 5,
        "total_tokens": 12,
        "completion_tokens": 7
      }
    }
    ```