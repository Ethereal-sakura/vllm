# KubeRay

[KubeRay](https://github.com/ray-project/kuberay) 提供了一种原生支持 Kubernetes 的方式，在 Ray 集群上运行 vLLM 工作负载。
你可以通过 YAML 声明一个 Ray 集群，KubeRay Operator 会自动处理 Pod 调度、网络配置、重启以及蓝绿部署——整个过程都符合 Kubernetes 的操作习惯。

## 为什么选择 KubeRay 而不是手动脚本？

| 功能 | 手动脚本 | KubeRay |
|---------|-----------------------------------------------------------|---------|
| 集群启动 | 需要手动 SSH 到每个节点并运行脚本 | 一条命令即可创建或更新整个集群：`kubectl apply -f cluster.yaml` |
| 自动扩缩容 | 需要手动操作 | 自动修改 CRD 来调整集群规模 |
| 升级 | 需要手动销毁并重建 | 支持蓝绿部署方案，升级更平滑 |
| 声明式配置 | 使用 Bash 参数和环境变量 | 适合 GitOps 的 YAML CRD（RayCluster/RayService） |

使用 KubeRay 可以显著降低运维负担，并让 Ray + vLLM 更容易与现有 Kubernetes 工作流（如 CI/CD、密钥管理、存储类等）集成。

## 进一步了解

* ["在 Kubernetes 上使用 Ray Serve LLM 部署大模型"](https://docs.ray.io/en/master/cluster/kubernetes/examples/rayserve-llm-example.html) —— 一个使用 vLLM、KubeRay 和 Ray Serve 实现模型部署的端到端示例
* [KubeRay 官方文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)