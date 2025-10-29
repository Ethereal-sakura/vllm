# P2P NCCL Connector

一个基于点对点通信的 xPyD 动态扩缩容实现，部分灵感来自 Dynamo。

## 详细设计

### 整体流程

如图 1 所示，本 **PD 反解耦** 方案的整体流程通过一次请求流转来描述：

1. 客户端向 Proxy/Router 的 `/v1/completions` 接口发送 HTTP 请求。
2. Proxy/Router 通过轮询或随机方式选择一个 **1P1D（1 个 Prefill 实例 + 1 个 Decode 实例）**，生成 `request_id`（生成规则后面会介绍），并将 HTTP 请求中的 `max_tokens` 字段修改为 **1**，然后转发给 **P 实例**。
3. 紧接着，Proxy/Router 会将**原始 HTTP 请求**也转发给 **D 实例**。
4. **P 实例**完成 **Prefill** 后，会**主动将生成的 KV 缓存**发送给 D 实例（采用 **PUT_ASYNC** 方式）。D 实例的 `zmq_addr` 可通过 `request_id` 解析获得。
5. **D 实例**有一条**专用线程**用于接收 KV 缓存（防止主进程阻塞）。接收到的 KV 缓存会存入 **GPU 内存缓冲区**，缓冲区大小由 vLLM 启动参数 `kv_buffer_size` 决定。当 GPU 缓冲区满时，KV 缓存会被存到**本地 Tensor 内存池**。
6. 在 **Decode** 阶段，D 实例主进程会从 **GPU 缓冲区**或**内存池**中获取 P 实例传来的 KV 缓存，从而**跳过 Prefill**。
7. **Decode** 完成后，D 实例将结果返回给 **Proxy/Router**，再由 Proxy/Router 转发给 **客户端**。

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

### Proxy/Router（Demo）

这是一个简单的 HTTP 服务，作为客户端请求的入口，并在后台启动线程监听 P/D 实例上报自己的 HTTP IP 和端口、ZMQ IP 和端口。它维护了一个 `http_addr -> zmq_addr` 的字典，其中 `http_addr` 是 vLLM 实例请求使用的 IP:PORT，`zmq_addr` 是进行 KV 缓存握手和元数据收发的地址。

Proxy/Router 会根据客户端请求特征（例如 prompt），选择 1P1D，并生成对应的 `request_id`，例如：

```text
cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```

目前为了快速验证 xPyD 能否跑通，选择 1P1D 时采用轮询。未来计划结合实例负载状态和前缀树方式，选择合适的 P 和 D。

每个 P/D 实例会定期（目前每 3 秒）向 Proxy/Router 发送心跳包进行注册（即上报 `http_addr -> zmq_addr`）并维持连接。如果某个实例崩溃并在一段时间内未发送心跳，Proxy/Router 会将该超时实例移除（此功能暂未开发）。

### KV 缓存传输方式

KVCache 支持三种传输方式：PUT、GET 和 PUT_ASYNC。可以通过 `--kv-transfer-config` 和 `kv_connector_extra_config` 参数指定，具体通过 `send_type` 字段选择。PUT 和 PUT_ASYNC 都是 P 实例主动发送 KVCache 给 D 实例，区别在于 PUT 是同步传输，主进程会阻塞；PUT_ASYNC 是异步传输，通过专用线程发送，不会阻塞主进程。GET 方式下，P 实例将 Prefill 后的 KVCache 存入内存缓冲区，D 实例在分配好 KVCache 空间后主动从 P 实例拉取。

实验结果显示三种方式的性能从高到低依次为：PUT_ASYNC → GET → PUT。

### ZMQ & NCCL 点对点通信

只要知道对方地址，就能实现点对点（P2P）的 KV 缓存传输（通过 NCCL），不受 rank 和 world size 限制，支持 PD 解耦场景下的实例动态扩缩容。也就是说，增加或减少 P/D 实例无需全系统重启。

每个 P/D 实例只需创建一个 `P2pNcclEngine` 实例。该实例会维护一个 ZMQ Server，专用线程监听 `zmq_addr`，接收来自其他实例的控制流请求——包括建立 NCCL 连接、发送 KVCache 元数据（如 tensor shape、dtype）等，但不负责实际的 KVCache 数据传输。

P 实例和 D 实例首次传输 KVCache 时，会建立一次 ZMQ 连接和 NCCL group。后续 KVCache 传输会复用这组连接。每个 NCCL group 仅包含两个 rank，即 world size = 2。这样的设计可支持动态扩缩容——只要知道对方地址，就能随时点对点发 KVCache，不受 rank/world size 限制，无需重启。

### NCCL Group 拓扑结构

目前仅支持对称 TP（张量并行 Tensor Parallelism）下的 KVCache 传输，后续会支持非对称 TP 和 PP（流水线并行 Pipeline Parallelism）。如图 2 所示，1P2D 场景下，每个实例的 TP 度为 2，共有 7 个 NCCL group：三台 vLLM 实例各有一个 TP=2 的 NCCL group。此外，P 实例的第 0 块 GPU 与每个 D 实例的第 0 块 GPU 各建立一个 NCCL group，第 1 块 GPU 同理。

![image2](https://github.com/user-attachments/assets/837e61d6-365e-4cbf-8640-6dd7ab295b36)

每个 NCCL group 都会占用一定的 GPU 内存缓冲区，大小主要受 `NCCL_MAX_NCHANNELS` 环境变量影响。比如 `NCCL_MAX_NCHANNELS=16` 时通常占用 100MB，`NCCL_MAX_NCHANNELS=8` 时一般占用 52MB。对于大规模 xPyD 配置（比如 DeepSeek 的 96P144D），目前实现还无法支持。后续计划采用 RDMA 实现点对点通信，也在关注 UCCL 方案。

### GPU 内存缓冲区与 Tensor 内存池

内存缓冲区大小的取舍如下：对于 P 实例，PUT 和 PUT_ASYNC 方式无需内存缓冲区，GET 方式需要。对于 D 实例，三种方式都需要内存缓冲区。D 实例的内存缓冲区不宜过大，P 实例在 GET 方式下内存缓冲区同理。D 实例的内存缓冲区用于暂存 P 实例发来的 KVCache，若设置过大，会挤占正常推理用的 KVCache 空间，降低推理批次，最终导致吞吐下降。内存缓冲区大小通过 `kv_buffer_size` 参数配置，单位为字节，通常设置为显存的 5%～10%。

如果 P 实例的 `--max-num-seqs` 参数设置过大，批次变大，会一次性产生大量 KVCache，可能超过 D 实例的内存缓冲区，导致 KVCache 丢失。KVCache 丢失后，D 实例需要重新 Prefill，相当于 Prefill 做了两次，TTFT（首次 token 延迟）急剧升高，性能恶化。

为了解决上述问题，我设计并实现了本地 Tensor 内存池用于存放 KVCache，其思想借鉴了 Linux 内核中的 buddy 内存分配系统。由于服务器端内存通常足够大（TB 级），无需考虑前缀缓存、块复用等节省内存的设计。内存缓冲区不足时，KVCache 直接存入 Tensor 内存池，D 实例可随后从中读取，读写速度为 PCIe，PCIe 4.0 大约 21 GB/s，通常比 Prefill 速度快，否则 Mooncake、lmcache 这类方案也不会存在。Tensor 内存池相当于“泄洪区”，平时不用，只在突发流量时启用。最坏情况下，性能也不会比普通 Cache store 差。

## 安装 vLLM

```shell
pip install "vllm>=0.9.2"
```

## 运行 xPyD

### 使用说明

- 以下示例在 A800 (80GB) 设备上运行，使用 Meta-Llama-3.1-8B-Instruct 模型。
- 请关注 `kv_buffer_size`（单位字节）的设置，经验值为显存的 10%。这个参数与 kvcache 大小有关，太小会导致临时存储 kvcache 的 GPU 缓冲区溢出，从而把 kvcache 存到 tensor 内存池，增加延迟；太大则会减少可用于推理的 kvcache 空间，批次变小，降低吞吐。
- Prefill 实例在非 GET 方式下，`kv_buffer_size` 可设置为 1，因为 Prefill 当前不需要接收 kvcache；若使用 GET 方式，则需要较大 `kv_buffer_size`，因为要存储发往 D 实例的 kvcache。
- 如有端口冲突，请自行调整下方命令中的 `kv_buffer_size` 和 `port`。
- 建议优先选择性能最好的 `PUT_ASYNC` 方式。
- `--port` 必须与 `--kv-transfer-config` 里的 `http_port` 保持一致。
- `disagg_proxy_p2p_nccl_xpyd.py` 脚本会使用 10001 端口（接收客户端请求）和 30001 端口（接收 P/D 实例服务发现）。
- 运行 proxy 的节点，需提前安装 `quart`。
- 支持多节点，只需修改 `--kv-transfer-config` 里的 `proxy_ip` 和 `proxy_port`。
- 以下示例默认 **proxy 的 IP 为 10.0.1.1**。

### 运行 1P3D

#### Proxy（如 10.0.1.1）

```shell
cd {your vllm directory}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

#### Prefill1（如 10.0.1.2 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode1（如 10.0.1.3 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode2（如 10.0.1.4 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode3（如 10.0.1.5 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20004 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' > /var/vllm.log 2>&1 &
    ```

### 运行 3P1D

#### Proxy（如 10.0.1.1）

```shell
cd {your vllm directory}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

#### Prefill1（如 10.0.1.2 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

#### Prefill2（如 10.0.1.3 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

#### Prefill3（如 10.0.1.4 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model