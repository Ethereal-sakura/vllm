# KubeAI

[KubeAI](https://github.com/substratusai/kubeai) 是一个 Kubernetes 运维工具（operator），可以让你在 Kubernetes 上轻松部署和管理 AI 模型。它为生产环境下部署 vLLM 提供了一种简单且可扩展的方式。诸如零起步扩容（scale-from-zero）、基于负载的自动扩缩容、模型缓存等功能都开箱即用，无需任何外部依赖。

请根据你的环境，参考安装指南进行安装：

- [任意 Kubernetes 集群](https://www.kubeai.org/installation/any/)
- [EKS](https://www.kubeai.org/installation/eks/)
- [GKE](https://www.kubeai.org/installation/gke/)

安装好 KubeAI 后，你可以通过 vLLM
[配置文本生成模型](https://www.kubeai.org/how-to/configure-text-generation-models/)
来实现 AI 能力。