# Anyscale

[Anyscale](https://www.anyscale.com) 是由 Ray 创始团队开发的一款托管型多云平台。

Anyscale 可以自动管理你在 AWS、GCP 或 Azure 账号中的 Ray 集群全生命周期，让你无需自己维护 Kubernetes 控制面、配置自动扩缩容、管理监控系统，也不需要用类似 [examples/online_serving/run_cluster.sh](../../../examples/online_serving/run_cluster.sh) 这样的辅助脚本手动操作主节点和工作节点，从而充分体验开源 Ray 的灵活性。

如果你使用 vLLM 部署大型语言模型，Anyscale 可以快速创建 [适用于生产环境的 HTTPS 端点](https://docs.anyscale.com/examples/deploy-ray-serve-llms) 或 [具备容错能力的批量推理任务](https://docs.anyscale.com/examples/ray-data-llm)。

## 在 Anyscale 上使用 vLLM 的生产级快速入门

- [离线批量推理](https://console.anyscale.com/template-preview/llm_batch_inference?utm_source=vllm_docs)
- [部署 vLLM 服务](https://console.anyscale.com/template-preview/llm_serving?utm_source=vllm_docs)
- [数据集整理](https://console.anyscale.com/template-preview/audio-dataset-curation-llm-judge?utm_source=vllm_docs)
- [微调 LLM](https://console.anyscale.com/template-preview/entity-recognition-with-llms?utm_source=vllm_docs)
