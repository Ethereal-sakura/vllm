# Google TPU

Tensor Processing Units（TPU，张量处理单元）是 Google 专门为加速机器学习工作负载而开发的定制应用专用集成电路（ASIC）芯片。TPU 有多个版本，每个版本的硬件配置各不相同。关于 TPU 的更多信息，请参考 [TPU 系统架构](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)  
关于 vLLM 支持的 TPU 版本，详情请见：

- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)

这些 TPU 版本支持对芯片的物理排列方式进行配置，从而提升吞吐量和网络性能。更多信息请参考：

- [TPU v6e 拓扑结构](https://cloud.google.com/tpu/docs/v6e#configurations)
- [TPU v5e 拓扑结构](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)
- [TPU v5p 拓扑结构](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)
- [TPU v4 拓扑结构](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

想要在 Google Cloud Platform 项目中使用 Cloud TPU，必须先为你的项目申请到相应的 TPU 配额。TPU 配额决定了你在 GCP 项目中可用的 TPU 数量，配额会根据 TPU 版本、数量和类型来分配。详细说明可查阅 [TPU 配额](https://cloud.google.com/tpu/docs/quota#tpu_quota)  

TPU 价格详情请参考 [Cloud TPU 定价](https://cloud.google.com/tpu/pricing)  

你可能还需要为 TPU 虚拟机（VM）配置额外的持久化存储，具体请参见 [Cloud TPU 数据存储选项](https://cloud.devsite.corp.google.com/tpu/docs/storage-options)  

!!! warning
    目前该设备没有预编译的 wheel 包，因此你需要使用官方提供的 Docker 镜像，或者自行从源码编译 vLLM。

## 环境要求

- Google Cloud TPU 虚拟机（VM）
- 支持的 TPU 版本：v6e、v5e、v5p、v4
- Python：3.11 或更高版本

### 申请 Cloud TPU

你可以通过 [Cloud TPU API](https://cloud.google.com/tpu/docs/reference/rest) 或 [queued resources](https://cloud.google.com/tpu/docs/queued-resources) API（推荐方式）来创建 Cloud TPU。本节以 queued resource API 为例，演示如何创建 TPU。更多关于 Cloud TPU API 的用法，请参考 [使用 Create Node API 创建 Cloud TPU](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api)  
Queued resources（排队资源）允许你以队列方式申请 Cloud TPU 资源。申请被提交后会进入 Cloud TPU 服务维护的队列中，当所需资源可用时，会自动分配给你的 Google Cloud 项目，供你专属使用。

!!! note
    在以下所有命令中，请将参数名（全部大写）替换为实际的值。具体参数说明请参考下表。

### 在 GKE 中使用 Cloud TPU

关于在 GKE（Google Kubernetes Engine）中使用 TPU 的更多信息，请参考：

- [GKE 中的 TPU 简介](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)
- [在 GKE Standard 部署 TPU 工作负载](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [GKE 中的 TPU 规划](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)

## 配置新环境

### 通过 queued resource API 申请 Cloud TPU

以创建 4 芯片的 TPU v5e 为例：

```bash
gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
  --node-id TPU_NAME \
  --project PROJECT_ID \
  --zone ZONE \
  --accelerator-type ACCELERATOR_TYPE \
  --runtime-version RUNTIME_VERSION \
  --service-account SERVICE_ACCOUNT
```

| 参数名             | 说明                                                                                                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| QUEUED_RESOURCE_ID | 用户自定义的排队资源请求 ID。                                                                                                                          |
| TPU_NAME           | 用户自定义的 TPU 名称，在分配资源请求后创建。                                                                                                          |
| PROJECT_ID         | 你的 Google Cloud 项目 ID。                                                                                                                           |
| ZONE               | 创建 Cloud TPU 的 GCP 区域。具体取值依赖于所选的 TPU 版本。详情请查阅 [TPU 区域和可用区]                                                               |
| ACCELERATOR_TYPE   | 你要使用的 TPU 版本。例如 `v5litepod-4` 表示 4 核心的 v5e TPU，`v6e-1` 表示 1 核心的 v6e TPU。更多信息见 [TPU 版本]                                 |
| RUNTIME_VERSION    | 需要使用的 TPU VM 运行环境版本。例如，`v2-alpha-tpuv6e` 表示适用于 v6e TPU 的虚拟机镜像。                                                              |
| SERVICE_ACCOUNT    | 服务账号的邮箱地址，可在 IAM Cloud Console 的 *Service Accounts* 页面找到。例如：`tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`        |

通过 SSH 连接到你的 TPU 虚拟机：

```bash
gcloud compute tpus tpu-vm ssh TPU_NAME --project PROJECT_ID --zone ZONE
```

!!! note
    在 GCP 上配置 `RUNTIME_VERSION`（“TPU 软件版本”）时，请务必参考 [TPU VM 镜像] 兼容性矩阵，确保所选版本与 TPU 代数一致。如果版本不兼容，vLLM 可能无法正常运行。

[TPU 版本]: https://cloud.google.com/tpu/docs/runtimes
[TPU VM 镜像]: https://cloud.google.com/tpu/docs/runtimes
[TPU 区域和可用区]: https://cloud.google.com/tpu/docs/regions-zones

## 使用 Python 进行环境搭建

### 预编译 wheel 包

目前官方尚未提供 TPU 的预编译 wheel 包。

### 源码构建 wheel 包

安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

创建并激活 vLLM 独立环境：

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

克隆 vLLM 仓库并进入目录：

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
```

卸载已有的 `torch` 和 `torch_xla`：

```bash
pip uninstall torch torch-xla -y
```

安装构建依赖：

```bash
pip install -r requirements/tpu.txt
sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
```

运行安装脚本：

```bash
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .
```

## 使用 Docker 进行环境搭建

### 预构建镜像

请参考 [使用 Docker](../../deployment/docker.md) 相关文档，使用官方 Docker 镜像时，请将镜像名 `vllm/vllm-openai` 替换为 `vllm/vllm-tpu`。

### 从源码构建镜像

你也可以使用 [docker/Dockerfile.tpu](../../../docker/Dockerfile.tpu) 构建支持 TPU 的 Docker 镜像。

```bash
docker build -f docker/Dockerfile.tpu -t vllm-tpu .
```

运行镜像时请使用如下命令：

```bash
# 请确保添加了 `--privileged --net host --shm-size=16G` 参数。
docker run --privileged --net host --shm-size=16G -it vllm-tpu
```

!!! note
    由于 TPU 依赖于 XLA，而 XLA 需要静态 shape，vLLM 会将可能输入 shape 进行分桶（bucketize），并为每种 shape 编译 XLA 计算图。首次编译可能需要 20~30 分钟。之后再次运行只需约 5 分钟，因为 XLA 计算图已缓存到磁盘（默认在 `VLLM_XLA_CACHE_PATH` 或 `~/.cache/vllm/xla_cache`）。

!!! tip
    如果遇到如下错误：

    ```console
    from torch._C import *  # noqa: F403
    ImportError: libopenblas.so.0: cannot open shared object file: No such
    file or directory
    ```

    可以使用以下命令安装 OpenBLAS 依赖：

    ```bash
    sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
    ```
