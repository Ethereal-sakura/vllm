# Helm

用于在 Kubernetes 上部署 vLLM 的 Helm Chart

Helm 是 Kubernetes 的一个包管理工具，可以帮助你自动化部署 vLLM 应用到 Kubernetes 集群中。通过 Helm，你可以在多个命名空间（namespace）下，根据不同需求覆盖变量值，部署统一的架构框架。

本指南将带你一步步完成使用 Helm 部署 vLLM 的流程，包括所需的前置条件、Helm 的安装步骤，以及架构和 values 文件的相关说明。

## 前置条件

在开始之前，请确保你已经具备以下条件：

- 一个正常运行的 Kubernetes 集群
- 已部署 NVIDIA Kubernetes Device Plugin（`k8s-device-plugin`）：获取地址 [https://github.com/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)
- 集群中有可用的 GPU 资源
- 一个包含待部署模型的 S3 存储

## 安装 Chart

以 `test-vllm` 作为发布名称安装 Chart 的命令如下：

```bash
helm upgrade --install --create-namespace \
  --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3endpoint=$ACCESS_POINT \
  --set secrets.s3bucketname=$BUCKET \
  --set secrets.s3accesskeyid=$ACCESS_KEY \
  --set secrets.s3accesskey=$SECRET_KEY
```

## 卸载 Chart

若需卸载 `test-vllm` 部署，可以执行：

```bash
helm uninstall test-vllm --namespace=ns-vllm
```

该命令会移除所有与该 Chart 相关的 Kubernetes 组件  
**包括持久化卷（persistent volumes）**，并删除对应的发布记录。

## 架构

![helm 部署架构](../../assets/deployment/architecture_helm_deployment.png)

## Values

下面的表格详细说明了 `values.yaml` 中可配置参数：

| Key | Type | Default | 说明 |
|-----|------|---------|-------------|
| autoscaling | object | {"enabled":false,"maxReplicas":100,"minReplicas":1,"targetCPUUtilizationPercentage":80} | 自动扩缩容相关配置 |
| autoscaling.enabled | bool | false | 是否启用自动扩缩容 |
| autoscaling.maxReplicas | int | 100 | 最大副本数 |
| autoscaling.minReplicas | int | 1 | 最小副本数 |
| autoscaling.targetCPUUtilizationPercentage | int | 80 | 用于自动扩缩容的目标 CPU 利用率 |
| configs | object | {} | ConfigMap 配置 |
| containerPort | int | 8000 | 容器端口 |
| customObjects | list | [] | 自定义对象相关配置 |
| deploymentStrategy | object | {} | 部署策略配置 |
| externalConfigs | list | [] | 外部配置 |
| extraContainers | list | [] | 额外容器的配置 |
| extraInit | object | {"pvcStorage":"1Gi","s3modelpath":"relative_s3_model_path/opt-125m", "awsEc2MetadataDisabled": true} | 初始化容器的额外配置 |
| extraInit.pvcStorage | string | "1Gi" | S3 挂载所需存储空间大小 |
| extraInit.s3modelpath | string | "relative_s3_model_path/opt-125m" | S3 上存放模型权重和配置文件的路径 |
| extraInit.awsEc2MetadataDisabled | boolean | true | 是否禁用 Amazon EC2 实例的元数据服务 |
| extraPorts | list | [] | 额外端口的配置 |
| gpuModels | list | ["TYPE_GPU_USED"] | 使用的 GPU 类型 |
| image | object | {"command":["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"],"repository":"vllm/vllm-openai","tag":"latest"} | 镜像相关配置 |
| image.command | list | ["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"] | 容器启动命令 |
| image.repository | string | "vllm/vllm-openai" | 镜像仓库 |
| image.tag | string | "latest" | 镜像标签 |
| livenessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":15,"periodSeconds":10} | 存活性探针配置 |
| livenessProbe.failureThreshold | int | 3 | 探针连续失败多少次后，Kubernetes 判定容器不健康 |
| livenessProbe.httpGet | object | {"path":"/health","port":8000} | kubelet 访问服务的 HTTP 请求配置 |
| livenessProbe.httpGet.path | string | "/health" | HTTP 服务的健康检查路径 |
| livenessProbe.httpGet.port | int | 8000 | 容器监听的端口名称或编号 |
| livenessProbe.initialDelaySeconds | int | 15 | 容器启动后多久开始执行存活性探针 |
| livenessProbe.periodSeconds | int | 10 | 存活性探针执行的时间间隔（秒） |
| maxUnavailablePodDisruptionBudget | string | "" | Pod 容忍中断相关配置 |
| readinessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":5,"periodSeconds":5} | 就绪性探针配置 |
| readinessProbe.failureThreshold | int | 3 | 探针连续失败多少次后，Kubernetes 判定容器未就绪 |
| readinessProbe.httpGet | object | {"path":"/health","port":8000} | kubelet 访问服务的 HTTP 请求配置 |
| readinessProbe.httpGet.path | string | "/health" | HTTP 服务的健康检查路径 |
| readinessProbe.httpGet.port | int | 8000 | 容器监听的端口名称或编号 |
| readinessProbe.initialDelaySeconds | int | 5 | 容器启动后多久开始执行就绪性探针 |
| readinessProbe.periodSeconds | int | 5 | 就绪性探针执行的时间间隔（秒） |
| replicaCount | int | 1 | 副本数量 |
| resources | object | {"limits":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1},"requests":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1}} | 资源限制和请求配置 |
| resources.limits."nvidia.com/gpu" | int | 1 | 使用的 GPU 数量 |
| resources.limits.cpu | int | 4 | 分配的 CPU 核数 |
| resources.limits.memory | string | "16Gi" | 分配的 CPU 内存 |
| resources.requests."nvidia.com/gpu" | int | 1 | 申请的 GPU 数量 |
| resources.requests.cpu | int | 4 | 申请的 CPU 核数 |
| resources.requests.memory | string | "16Gi" | 申请的内存大小 |
| secrets | object | {} | 密钥相关配置 |
| serviceName | string | "" | 服务名称 |
| servicePort | int | 80 | 服务端口 |
| labels.environment | string | test | 环境名称 |
