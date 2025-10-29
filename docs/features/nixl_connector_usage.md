# NixlConnector 使用指南

NixlConnector 是一个专为 vLLM 的分离式预填充（disaggregated prefilling）特性设计的高性能 KV 缓存传输连接器。它通过 NIXL 库实现跨进程 KV 缓存的高效异步收发操作。

## 使用准备

### 安装

快速开始：安装 NIXL 库，可以直接运行 `uv pip install nixl`。

- 更多安装步骤请查阅 [NIXL 官方仓库](https://github.com/ai-dynamo/nixl) 
- 需要的 NIXL 版本在 [requirements/kv_connectors.txt](../../requirements/kv_connectors.txt) 及相关配置文件中有说明

如果是在非 CUDA 平台，请按照下面方式用 UCX 从源码安装 nixl。

```bash
python tools/install_nixl_from_source_ubuntu.py
```

### 传输配置

NixlConnector 底层通信采用 NIXL 库，支持多种传输后端。UCX（Unified Communication X）是 NIXL 的默认主用传输库。你可以通过环境变量配置传输方式：

```bash
# UCX 配置示例，请根据实际环境调整
export UCX_TLS=all  # 或者指定具体传输，例如 "rc,ud,sm,^cuda_ipc" 等
export UCX_NET_DEVICES=all  # 或者指定网卡设备，例如 "mlx5_0:1,mlx5_1:1"
```

!!! tip
    当使用 UCX 作为传输后端时，NCCL 的环境变量（如 `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`）对 NixlConnector 不生效。请只配置 UCX 相关环境变量，不必设置 NCCL 变量。

## 基础用法（同一台主机）

### Producer（预填充）配置

启动一个预填充实例，生成 KV 缓存：

```bash
# 第一块 GPU 作为预填充
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8100 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### Consumer（解码）配置

启动一个解码实例，消费 KV 缓存：

```bash
# 第二块 GPU 作为解码
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8200 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### 代理服务器

通过代理服务器在预填充和解码之间转发请求：

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts localhost \
  --prefiller-ports 8100 \
  --decoder-hosts localhost \
  --decoder-ports 8200
```

## 环境变量说明

- `VLLM_NIXL_SIDE_CHANNEL_PORT`：NIXL 握手通信端口
    - 默认值：5600
    - **预填充和解码实例都必须设置**
    - 每个 vLLM worker 在同一主机上需分配不同端口，不同主机间端口号可以相同
    - TP/DP 部署时，每个节点 worker 的端口为：base_port + dp_rank * tp_size + tp_rank（比如 `--tensor-parallel-size=4` 且 base_port=5600，则 tp_rank 0..3 分别用 5600, 5601, 5602, 5603）
    - 该端口用于预填充与解码之间的 NIXL 握手连接

- `VLLM_NIXL_SIDE_CHANNEL_HOST`：侧信道通信的主机地址
    - 默认值："localhost"
    - 如果预填充和解码不在同一台机器，需要设置为对方 IP
    - 握手的连接信息会通过 KVTransferParams 从预填充传递到解码端

- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`：自动释放预填充 KV 缓存的超时时间（单位秒，可选）
    - 默认值：480
    - 如果某个请求被中止且解码端还未通过 nixl 通道读取 KV 缓存块，超时后预填充实例将主动释放这些缓存，避免资源被长期占用

## 多实例部署

### 多台机器上的多个预填充实例

```bash
# 预填充 1，部署在机器 A（示例 IP: ${IP1}）
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP1} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# 预填充 2，部署在机器 B（示例 IP: ${IP2}）
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

### 多台机器上的多个解码实例

```bash
# 解码 1，部署在机器 C（示例 IP: ${IP3}）
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP3} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# 解码 2，部署在机器 D（示例 IP: ${IP4}）
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP4} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

### 多实例场景的代理配置

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts ${IP1} ${IP2} \
  --prefiller-ports 8000 8000 \
  --decoder-hosts ${IP3} ${IP4} \
  --decoder-ports 8000 8000
```

### KV 角色选项说明

- **kv_producer**：预填充实例，负责生成 KV 缓存
- **kv_consumer**：解码实例，从预填充端获取 KV 缓存
- **kv_both**：允许连接器同时作为生产者和消费者，方便实验或角色未确定的场景使用

!!! tip
    NixlConnector 当前不会区分 `kv_role`，实际的预填充/解码角色由上层代理（例如 `toy_proxy_server.py` 的 `--prefiller-hosts` 和 `--decoder-hosts` 参数）决定。
    所以，在 `--kv-transfer-config` 里设置的 `kv_role` 只是一个占位符，并不会影响 NixlConnector 的行为。

## 实验性功能

### 支持异构 KV 缓存布局

应用场景：使用 'HND' 进行预填充，使用 'NHD' 进行解码，通过实验性配置支持此需求

```bash
--kv-transfer-config '{..., "enable_permute_local_kv":"True"}'
```

## 示例脚本/代码

你可以在 vLLM 仓库中找到以下参考脚本：

- [run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh)
- [toy_proxy_server.py](../../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py)
- [test_accuracy.py](../../tests/v1/kv_connector/nixl_integration/test_accuracy.py)
