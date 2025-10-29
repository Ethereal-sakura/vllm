# 环境变量

vLLM 通过以下环境变量来配置系统：

!!! warning
    请注意，`VLLM_PORT` 和 `VLLM_HOST_IP` 是为 vLLM 的**内部使用**设置端口和 IP 的。它们不是 API 服务器的端口和 IP。如果你用 `--host $VLLM_HOST_IP` 和 `--port $VLLM_PORT` 启动 API 服务器，是无法正常工作的。

    vLLM 使用的所有环境变量，都以 `VLLM_` 为前缀。**Kubernetes 用户特别注意**：请不要将服务命名为 `vllm`，否则 Kubernetes 设置的环境变量可能会与 vLLM 的环境变量发生冲突。因为 [Kubernetes 会为每个服务设置以服务名大写为前缀的环境变量](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables) 

??? code

    ```python
    --8<-- "vllm/envs.py:env-vars-definition"
    ```