# 分布式部署故障排查

如需通用故障排查，请参考 [故障排查](../usage/troubleshooting.md)。

## 验证节点间的 GPU 通信

启动 Ray 集群后，请务必检查各节点之间的 GPU 到 GPU 的通信是否正常。正确配置并不总是容易的。详细说明可见 [故障排查脚本](../usage/troubleshooting.md#incorrect-hardwaredriver)。如果 GPU 通信需要额外的环境变量配置，可以把这些变量添加到 [examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh) 文件中，例如 `-e NCCL_SOCKET_IFNAME=eth0`。建议在集群创建阶段设置环境变量，这样可以确保变量同步到所有节点。相反，如果只在本地 shell 设置环境变量，则只对当前节点生效。更多细节可参考 <https://github.com/vllm-project/vllm/issues/6803> 

## 没有可用的节点类型满足资源请求

即使集群中有足够的 GPU，有时也会遇到 `Error: No available node types can fulfill resource request` 这样的问题。这通常是因为节点有多个 IP 地址，而 vLLM 未能正确选择需要的 IP。请通过在 [examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh) 中设置 `VLLM_HOST_IP`（每个节点应设置不同的值），确保 vLLM 和 Ray 使用的是同一个 IP 地址。你可以通过 `ray status` 和 `ray list nodes` 命令确认实际选用的 IP。详细内容可参见 <https://github.com/vllm-project/vllm/issues/7815> 

## Ray 可观测性（Observability）

分布式系统由于规模大且架构复杂，调试起来常常非常具有挑战性。Ray 提供了一整套工具，便于你监控、调试及优化 Ray 应用和集群。关于 Ray 可观测性的更多信息，请访问 [官方 Ray 可观测性文档](https://docs.ray.io/en/latest/ray-observability/index.html)。如果你需要调试 Ray 应用，可以参阅 [Ray 调试指南](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html)。如需排查 Kubernetes 集群相关问题，可参考 [官方 KubeRay 故障排查指南](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html)。
