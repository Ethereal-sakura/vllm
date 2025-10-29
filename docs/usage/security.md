# 安全性

## 节点间通信

在多节点 vLLM 部署中，节点之间的所有通信**默认都是不安全的**，必须通过将节点置于隔离网络中来进行保护。包括以下几类通信：

1. PyTorch 分布式通信
2. KV 缓存传输通信
3. 张量、流水线及数据并行通信

### 节点间通信的配置选项

以下配置项用于控制 vLLM 的节点间通信：

#### 1. **环境变量：**

- `VLLM_HOST_IP`：指定 vLLM 进程用于通信的 IP 地址

#### 2. **KV 缓存传输配置：**

- `--kv-ip`：KV 缓存传输通信的 IP 地址（默认值：127.0.0.1）
- `--kv-port`：KV 缓存传输通信的端口（默认值：14579）

#### 3. **数据并行配置：**

- `data_parallel_master_ip`：数据并行主节点的 IP（默认值：127.0.0.1）
- `data_parallel_master_port`：数据并行主节点的端口（默认值：29500）

### PyTorch 分布式相关说明

vLLM 部分节点间通信采用了 PyTorch 的分布式特性。关于 PyTorch 分布式安全性的详细信息，请参考 [PyTorch 安全指南](https://github.com/pytorch/pytorch/security/policy#using-distributed-features)

PyTorch 安全指南中的要点总结：

- PyTorch 分布式功能仅面向内部通信
- 不适合在不可信环境或网络中使用
- 为了性能考虑，未内置授权协议
- 消息传输未加密
- 任意来源的连接都会被接受，不做校验

### 安全性建议

#### 1. **网络隔离：**

- 将 vLLM 节点部署到专用、隔离的网络环境中
- 通过网络分段阻止未授权访问
- 配置合适的防火墙规则

#### 2. **配置最佳实践：**

- 始终为 `VLLM_HOST_IP` 设置明确的 IP 地址，避免使用默认值
- 防火墙设置仅允许节点间所需端口的通信

#### 3. **访问控制：**

- 限制部署环境的物理和网络访问权限
- 管理接口应实施严格的身份验证与授权
- 所有系统组件应遵循最小权限原则

### 4. **限制媒体 URL 域名访问：**

可以通过设置 `--allowed-media-domains` 限定 vLLM 能访问的媒体 URL 域名，从而防止服务器端请求伪造（SSRF）攻击。
（例如：`--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`）

此外，建议将 `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`，避免通过 HTTP 重定向绕过域名限制。

## 安全与防火墙：保护暴露的 vLLM 系统

虽然 vLLM 设计时就要求将不安全的网络服务隔离在私有网络中，但部分依赖组件和底层框架可能会启动监听所有网络接口的不安全服务，这往往不受 vLLM 直接控制。

尤其需要注意的是 `torch.distributed` 的使用，vLLM 利用它进行分布式通信，即便是在单机环境下使用 vLLM 时也会涉及。当 vLLM 采用 TCP 初始化时（详见 [PyTorch TCP 初始化文档](https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization)），PyTorch 会创建一个默认监听所有网络接口的 `TCPStore`。这意味着如果没有额外防护措施，任何能访问你主机的设备都可以连接这些服务。

**从 PyTorch 的角度来看，任何 `torch.distributed` 的使用默认都是不安全的。** 这是 PyTorch 团队有意为之的设计。

### 防火墙配置建议

保护 vLLM 系统的最佳方式是合理配置防火墙，只开放必要的网络端口。通常包括：

- **阻止所有入站连接，仅允许 API 服务监听的 TCP 端口对外开放**

- 确保用于内部通信（如 `torch.distributed` 和 KV 缓存传输）的端口仅对可信主机或网络开放

- 绝不将这些内部端口暴露给公网或不可信网络

具体的防火墙配置方法请参考你的操作系统或应用平台官方文档。

## 安全漏洞报告

如果你认为发现了 vLLM 的安全漏洞，请按照项目的安全政策进行报告。关于如何报告安全问题以及项目相关政策，请参阅 [vLLM 安全政策](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)
