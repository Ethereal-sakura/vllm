# CPU

vLLM 是一个 Python 库，支持以下几种 CPU 类型。请选择你的 CPU 类型，查看对应厂商的安装说明：

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:installation"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:installation"

## 环境要求

- Python：3.10 到 3.13

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:requirements"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:requirements"

## 使用 Python 环境部署

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### 预编译的 wheels 包

目前尚未提供 CPU 平台的预编译 wheels 包。

### 从源代码构建 wheel

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-wheel-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-wheel-from-source"

## 使用 Docker 部署

### 预构建镜像

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-images"

### 从源码构建镜像

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-image-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-image-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-image-from-source"

=== "IBM Z (S390X)"
    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-image-from-source"

## 相关运行环境变量

- `VLLM_CPU_KVCACHE_SPACE`：用于设置 KV Cache（键值缓存）的空间大小（比如 `VLLM_CPU_KVCACHE_SPACE=40` 表示分配 40 GiB 的 KV cache 空间），设置越大可以同时处理更多请求。具体值建议根据硬件配置以及内存管理方式调整。默认值为 `0`。
- `VLLM_CPU_OMP_THREADS_BIND`：指定哪些 CPU 核心专用于 OpenMP 线程，可以设置为 CPU id 列表或 `auto`（默认）。例如，`VLLM_CPU_OMP_THREADS_BIND=0-31` 表示会有 32 个 OpenMP 线程绑定在 0 到 31 号 CPU 核心上。`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` 表示启动 2 个 tensor 并行进程，rank0 的 32 个 OpenMP 线程绑定在 0-31 号 CPU 核心，rank1 的线程绑定在 32-63 号。若设置为 `auto`，则每个进程的 OpenMP 线程会自动绑定到各自的 NUMA 节点。
- `VLLM_CPU_NUM_OF_RESERVED_CPU`：为每个进程预留不参与 OpenMP 线程绑定的 CPU 核心数量。仅在 VLLM_CPU_OMP_THREADS_BIND 设置为 `auto` 时生效，默认值为 `None`。如果未设置且使用 `auto`，当 world_size == 1 不预留 CPU，world_size > 1 时每个进程预留 1 个 CPU。
- `CPU_VISIBLE_MEMORY_NODES`：指定 vLLM CPU worker 可见的 NUMA 内存节点，类似于 ```CUDA_VISIBLE_DEVICES```。该变量仅在 VLLM_CPU_OMP_THREADS_BIND 为 `auto` 时生效，可以用于屏蔽节点或调整绑定顺序，实现更灵活的自动线程绑定。
- `VLLM_CPU_MOE_PREPACK`（仅 x86）：是否对 MoE 层使用预打包（prepack），会传递给 `ipex.llm.modules.GatedMLPMOE`。默认值为 `1`（开启）。若 CPU 不支持，可能需要设置为 `0`（关闭）。
- `VLLM_CPU_SGL_KERNEL`（仅 x86，实验性功能）：是否对线性层和 MoE 层采用小批量优化内核，特别适合低延迟场景如在线服务。要求 CPU 支持 AMX 指令集、权重为 BFloat16 且形状为 32 的倍数。默认值为 `0`（关闭）。

## 常见问题 FAQ

### 应该使用哪个 `dtype`？

- 当前 vLLM CPU 默认采用模型自带的 dtype。但由于 torch CPU 的 float16 支持不稳定，若出现性能或精度问题，建议显式设置 `dtype=bfloat16`。

### 如何在 CPU 上启动 vLLM 服务？

- 使用在线服务时，建议为服务框架预留 1-2 个 CPU 核心，避免 CPU 资源过载。比如在拥有 32 个物理核心的平台上，可以预留 31 号 CPU 给服务框架，0-30 号 CPU 用于推理线程：

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16
```

或者使用默认的自动线程绑定：

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

注意，当 world_size == 1 时，建议手动为 vLLM 前端进程预留 1 个 CPU。

### 如何确定 `VLLM_CPU_OMP_THREADS_BIND` 的设置？

- 大多数情况下推荐使用默认的 `auto` 自动线程绑定。理想情况下，每个 OpenMP 线程绑定到一个独立物理核心，同一个进程的线程绑定到同一个 NUMA 节点，world_size > 1 时每个进程预留 1 个 CPU。若遇到性能问题或绑定异常，可参照以下方式手动绑定线程。

- 在启用超线程的 16 逻辑核 / 8 物理核平台上：

??? console "命令示例"

    ```console
    $ lscpu -e # 查看逻辑核与物理核的对应关系

    # "CPU" 列为逻辑核编号，"CORE" 列为物理核编号。此平台每两个逻辑核共享一个物理核。
    CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
    0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
    8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000

    # 在该平台上，建议只绑定 OpenMP 线程到 0-7 或 8-15 号逻辑核
    $ export VLLM_CPU_OMP_THREADS_BIND=0-7
    $ python examples/offline_inference/basic/basic.py
    ```

- 当在多路服务器（多 socket、NUMA）上部署 vLLM CPU 后端，并启用 tensor 并行或 pipeline 并行时，每个 NUMA 节点会作为一个 TP/PP 进程。请确保同一个进程的 CPU 核心都在同一个 NUMA 节点，避免跨 NUMA 节点的内存访问。

### 如何设置 `VLLM_CPU_KVCACHE_SPACE`？

该参数默认值为 4GB。更大的 KV cache 空间可以支持更多并发请求和更长的上下文，但需要注意每个 NUMA 节点的内存容量。每个 TP 进程的内存占用为 `weight shard size` 与 `VLLM_CPU_KVCACHE_SPACE` 之和，若超过单个 NUMA 节点的容量，TP worker 会因内存不足被 kill（exitcode 9）。

### 如何对 vLLM CPU 进行性能调优？

首先，请确保线程绑定和 KV cache 空间已正确设置并生效。可以通过运行 vLLM 基准测试并用 `htop` 观察 CPU 核心使用情况确认。

推理批大小（batch size）对性能影响很大。批越大吞吐量越高，批越小延迟越低。建议从默认值开始调整最大批大小，在吞吐和延迟之间找到平衡点，以提升 vLLM CPU 在特定平台上的性能。vLLM 有两个重要相关参数：

- `--max-num-batched-tokens`：限制单批次的 token 数量，对首 token 性能影响较大。默认值如下：
    - 离线推理：`4096 * world_size`
    - 在线服务：`2048 * world_size`
- `--max-num-seqs`：限制单批次序列数量，对输出 token 性能影响较大。
    - 离线推理：`256 * world_size`
    - 在线服务：`128 * world_size`

vLLM CPU 支持数据并行（DP）、张量并行（TP）和流水线并行（PP），可充分利用多 socket 和多内存节点。更多 DP、TP、PP 的调优细节请参见[优化与调优](../../configuration/optimization.md)。如有充足的 CPU socket 和内存节点，建议同时使用 DP、TP 和 PP。

### vLLM CPU 支持哪些量化配置？

- vLLM CPU 支持如下量化方式：
    - AWQ（仅 x86）
    - GPTQ（仅 x86）
    - compressed-tensor INT8 W8A8（x86、s390x）

### （仅 x86）`VLLM_CPU_MOE_PREPACK` 和 `VLLM_CPU_SGL_KERNEL` 有什么作用？

- 两者都需要 CPU 支持 `amx` 指令集。
    - `VLLM_CPU_MOE_PREPACK` 可提升 MoE 模型性能
    - `VLLM_CPU_SGL_KERNEL` 对 MoE 模型和小批量场景有更好的性能表现

### 为什么在 Docker 里运行时看到 `get_mempolicy: Operation not permitted`？

在部分容器环境（如 Docker）中，vLLM 用到的 NUMA 相关系统调用（如 `get_mempolicy`、`migrate_pages`）默认会被 seccomp/capabilities 限制，导致出现 `get_mempolicy: Operation not permitted` 的警告。功能不会受影响，但 NUMA 内存绑定和迁移优化无法生效，性能可能不理想。

要在 Docker 里开启这些优化，并尽量减少权限提升，可以参考以下设置：

```bash
docker run ... --cap-add SYS_NICE --security-opt seccomp=unconfined  ...

# 1) `--cap-add SYS_NICE` 用于解决 `get_mempolicy` 权限问题

# 2) `--security-opt seccomp=unconfined` 用于支持 `migrate_pages`，使 `numa_migrate_pages()` 正常工作。
# 实际上，`seccomp=unconfined` 会跳过容器 seccomp 限制，
# 如果不能接受，可以自定义 seccomp 配置文件，
# 基于 docker/runtime 默认的 default.json，将 `migrate_pages` 加入 `SCMP_ACT_ALLOW` 列表。

# 参考文档： https://docs.docker.com/engine/security/seccomp/
```

另外，也可以使用 `--privileged=true` 运行容器，但该方式权限提升较多，一般不推荐。

如果在 K8S 环境部署，可以在 workload 的 yaml 配置中添加以下内容，达到同样效果：

```yaml
securityContext:
  seccompProfile:
    type: Unconfined
  capabilities:
    add:
    - SYS_NICE
```