# 混合型 KV 缓存管理器

!!! warning
    本文档基于 [458e74](https://github.com/vllm-project/vllm/commit/458e74eb907f96069e6d8a4f3c9f457001fef2ea) 版本撰写。该特性仍处于早期阶段，未来可能会有变化。

## 什么是混合模型？

许多最新的“混合型”大语言模型（LLM）在一个模型中结合了多种注意力机制。例如：

1. 滑动窗口注意力（sliding window attention，sw） + 全局注意力（full attention）：gpt-oss、Gemma 2/3、Ministral、cohere 等。
2. Mamba + 全局注意力：Bamba、Jamba、Minimax 等。
3. 局部分块注意力（local chunked attention）+ 全局注意力：Llama4

为了高效服务这些模型，我们的 [KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 需要：

1. 为不同类型的层分配不同的缓存槽，例如：
    - 全局注意力层：为**所有** token 预留缓存槽。
    - 滑动窗口层：只为最近的 **`sliding_window_size`** 个 token 预留缓存槽。
2. 支持针对不同层的前缀缓存规则，例如：
    - 全局注意力：前缀命中要求 KV 缓存中保留**所有** token。
    - 滑动窗口：前缀命中只要求 KV 缓存中保留最后 **`sliding_window_size`** 个 token。

## 概念说明

1. **kv hidden size**：每一层存储一个 token 的 KV 缓存所需的字节数。
2. **block**：用于 KV 缓存的内存会被划分为多个 *blocks*，每个块都有相同的 *page size*（见下文定义）。
3. **block size**：一个 block 内包含的 token 数量。
4. **page size**：一个 block 的物理内存大小，定义如下：

    $$
    \text{num_layers} \times \text{block_size} \times \text{kv_hidden_size}
    $$

    其中 `num_layers` 并不一定指模型的总层数，具体数值需结合本文档上下文理解。

    !!! note
        这与代码里的 `KVCacheSpec.page_size_bytes` 定义不同，后者为：

        $$
        \text{block_size} \times \text{kv_hidden_size}
        $$

## 分配策略

### 高层思路

我们为所有层类型使用同一个内存池。内存池被划分为多个具有相同 page size 的 block。[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 会根据注意力类型为不同层分配不同数量的 block。

核心挑战在于，如何确保所有层类型使用相同的 **page size**。对于仅包含全局注意力的模型，page size 的定义很简单：

$$
\text{page_size} = \text{block_size} \times \text{num_hidden_layers} \times \text{kv_hidden_size}
$$

但在混合模型中，`num_hidden_layers` 会因注意力类型不同而变化，导致 page size 不一致。下列案例展示了我们如何统一 page size。

### 案例1：玩具模型

我们先来看一个简单示例：模型包含 1 层全局注意力和 3 层滑动窗口注意力，所有层的 `kv_hidden_size` 相同。

我们让每个 block 只存储一个层的 `block_size` 个 token，因此：

$$
\text{page_size} = \text{kv_hidden_size} \times \text{block_size}
$$

[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 会为每个层分配不同数量的 block。

这个只是玩具案例，真实模型请参考后续案例。

### 案例2：`kv_hidden_size` 相同且层分布有规律

如果模型包含更多层，比如 20 层滑动窗口注意力和 10 层全局注意力，并且 `kv_hidden_size` 都相同。每层分别调用一次分配器（共 30 次）虽然可行，但效率不高。为此，我们将需要相同 block 数量的层进行分组，减少分配次数。

之所以能分组，是因为不同类型层的数量通常有一定比例。例如：

- Gemma-2：1 个 sw 对 1 个 full
- Llama 4：3 个 local 对 1 个 full

我们的例子可以看作 2 个 sw 对 1 个 full。我们按 2 个 sw 和 1 个 full 的比例分配 block，然后将结果重复 10 次，生成 30 层的 `block_ids`。此时 page size 为：

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

假设 `block_size` 为 16，滑动窗口大小为 32，请求长度为 112，那么该模型需分配 11 个 block（0-6 给 full，7-8 给 sw 组1，9-10 给 sw 组2）。

![分组分配示例](../assets/design/hybrid_kv_cache_manager/basic_grouping_example.png)

图中，"/" 表示不用分配 block（滑动窗口层无需为早期 token 分配槽）。

正式定义如下。所有层会被分为多个 *KV Cache Group*，满足以下条件：

1. **组内注意力类型一致**：每组只包含相同注意力类型的层，这样同组内层可共享 block id，无需浪费内存。
2. **组间 page size 相同**：因为内存池只有一个 page size。

本例模型被分为 3 个 KV cache group：

- 第0组：10 层全局注意力（full.0 - full.9）
- 第1组：10 层滑动窗口注意力（sw.0 - sw.9）
- 第2组：10 层滑动窗口注意力（sw.10 - sw.19）

很明显满足以上两条规则。每组的 page size 都是：

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

### 案例3：`kv_hidden_size` 相同但层分布不规律

并非所有模型都能有如此理想的分组比例，案例2的方法会产生太多小组。例如，Gemma-3-27b 有 52 层滑动窗口注意力和 10 层全局注意力。用案例2的策略，会得到 26 个滑动窗口组和 5 个全局注意力组，每组仅含2层，分配效率仍不高。为此，我们按所有注意力类型中最小层数进行分组。例如，Gemma-3-27b 取 min(52, 10)=10，每组含10层，分组结果如下：

- 第0组：10 层全局注意力（full.0 - full.9）
- 第1组：10 层滑动窗口注意力（sw.0 - sw.9）
- 第2组：10 层滑动窗口注意力（sw.10 - sw.19）
- ...
- 第6组：10 层滑动窗口注意力（sw.40 - sw.49）
- 第7组：2 层滑动窗口注意力（sw.50 - sw.51）+ 8 个填充层

如果新模型出现（如 20 full + 30 sw），分组大小应调整为 10，而不是 20。我们会根据实际情况优化分组算法。

此策略常见于 Gemma-3 系列，以及部分案例2模型配合 eagle speculative decoding（会多出一层全局注意力）。这种分组会有一定内存浪费，不是最优方案。如果遇到填充导致内存浪费严重的情况，请反馈，我们会进一步优化算法。

### 案例4：`kv_hidden_size` 不同（主要针对混合型 Mamba 模型）

一些架构（如 Bamba、Jamba、Minimax）将标准注意力层与 Mamba 层交错使用，而 Mamba 层每个 token 的状态大小往往远大于注意力层的 `kv_hidden_size`。由于所有组都必须使用统一的 page size，我们需要协调不同的 hidden size。

当前算法为：

1. 增加注意力层的 `block_size`，直到满足
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \ge \text{state_size}_{\text{mamba}}
    $$
2. 对每层 Mamba 状态进行填充，达到
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}}
    $$
3. 按案例3的分组策略分组。

!!! note
    这种策略可能导致注意力层的 `block_size` 超过400，过大。另一种填充思路是：

    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \times \text{num_attn_layers} \ge \text{state_size}_{\text{mamba}}
    $$

    该策略仍在优化中。

### 案例5：KV 共享

KV 共享是指某个层直接复用另一个层的 KV 缓存，例如 gemma-3n。
在这些模型中，[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 会跳过所有 KV 共享的层，仅为真正需要 KV 缓存的层分配 KV，模型执行器中会做一些特殊处理以应用分配结果到 KV 共享层。

## 前缀缓存机制

为便于说明，本节假定 `block_size=1`。

### 基本思路

block 池采用类似 `tuple(block_hash, group_id) -> block` 的字典结构来捕获完整的 block。也就是说，不同组的相同 token 会独立缓存和清理。

当有新请求时，我们会检查每个组的前缀命中情况，并返回这些组的交集作为该请求的已缓存前缀。下文详细介绍了检查单组缓存命中的算法和交集操作。

### 案例0：仅含全局注意力模型

对于全局注意力层，请求的每个 token 都会分配 block。底层设计详见 [Prefix Caching](prefix_caching.md)

要找到请求的最长前缀缓存命中，我们从左到右依次检查每个 block 是否已缓存，遇到未命中后立即退出。例如，下图（蓝色 block 已缓存）会返回前7个 token（0-6）作为缓存命中前缀：

![全局注意力前缀缓存](../assets/design/hybrid_kv_cache_manager/full_attn.png)

### 案例1：仅含滑动窗口注意力模型

对于滑动窗口注意力层，如果采用最简单的分配方式，会分配 `sliding_window_size` 个 block，并采用轮询方式填充。但这种方式不兼容前缀缓存，因此我们并未采用。在 vLLM 中，每个 token 分配不同的 block，超出滑动窗口范围的 block 会被释放。

对于新请求，前缀命中只要求最后 `sliding_window_size - 1` 个 token 被缓存。
假设 `sliding_window_size = 4`，`block_size = 1`，请求长度为15（蓝色 block 已缓存）：

![滑动窗口注意力前缀缓存](../assets/design/hybrid_kv_cache_manager/sw_attn.png)

此时有3种可能的前缀命中：

- 命中长度5，预填充计算 [2, 3, 4] → [5, 6, …, 14]
- 命中长度6，预填充计算 [3, 4, 5] → [6, 7, …, 14]
- 命中长度14，预填充计算 [11, 12, 13] → [14]（效率最高）

我们从右向左检查缓存命中，找到后即可退出。与全局注意力（从左到右检查，遇到未命中即退出）正好相反。相比全局注意力，这种方式在未命中时要遍历全部 token，可能带来一定开销，不过混合 full + swa 的场景下影响较小，详见下文。

### 案例2：滑动窗口注意力 + 全局注意力模型

首先要解决的是如何找到缓存命中前缀。我们需要将全局注意力和滑动窗口注意力的缓存命中做“交集”：

1. 获取全局注意力的最长缓存命中（从左到右扫描）。
2. 在该长度范围内，获取滑动窗口注意力的最长缓存命中。具体做法是，从全局注意力命中长度起，右向左检查滑动窗口缓存命中。

这样能保证滑动窗口注意力层的缓存命中前缀，必定也是全局注意力层的缓存命中前缀。比起分别枚举每组所有可能前缀再做交集，我们的方法能早退出，提高效率。

该算法适用于恰好包含两种注意力类型的模型（全局注意力 + X），其中 X 可以是滑动窗口、llama 4 局部注意力、Mamba 等任意高效注意力机制。不支持没有全局注意力层或超过两种注意力类型的模型。目前主流混合模型均可满足。

第二个问题是缓存淘汰策略。当前所有 kv cache group 共享一个 LRU 队列。block 被释放（如请求结束或 block 超出滑动窗口范围）时会加入 LRU 队列。

### 案例3：Mamba 模型

Mamba 模型的前缀缓存支持尚在开发中。后续实现后，将可通过案例2的全局注意力 + X 算法支持 mamba + 全局注意力的模型。

## 实现说明

### 整体结构

![混合型 KV 缓存管理器结构总览](../assets/design/hybrid_kv_cache_manager/overview.png)

`KVCacheManager` 分为三层：

- **[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager]**：调度器和 KV 缓存管理之间的接口。
- **[KVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator]**：协调各组 SingleTypeKVCacheManagers，生成请求的分配结果。根据模型配置，选择以下协调器之一：
    - **[KVCacheCoordinatorNoPrefixCache][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinatorNoPrefixCache]**：禁用前缀缓存时使用。
    - **[UnitaryKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.UnitaryKVCacheCoordinator]**：仅有一个 KV cache group 时使用。前缀缓存逻辑简化，无需交集。
    - **[HybridKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator]**：处理恰好有两个 KV cache group（必须有一个全局注意力组和一个其他高效注意力组）。其他情况尚未实现，如需使用可禁用前缀缓存，采用 KVCacheCoordinatorNoPrefixCache。
- **[SingleTypeKVCacheManager][vllm.v1.core.single_type_kv_cache_manager.SingleTypeKVCacheManager]**：每个实例负责管理一个 KV cache group 的分配和前缀缓存，实现对应注意力类型的逻辑（如全局注意力、滑动窗口、Mamba）。

上图蓝色区域展示了 10 个全局注意力层和 20 个滑动窗口层的场景：

- 使用 `HybridKVCacheCoordinator`
- 3 个 `KVCacheGroup` 分别用 1 个 `FullAttentionManager` 和 2 个 `SlidingWindowManager`

### 内存布局

对于具有 n 个 `KVCacheGroup` 且每组有 m 层的模型，我们会分配 m 个 buffer。每个 buffer 被 n 个层共享，各组各一层。

下图展示了一个含 10 层全局注意力（full.0 - full.9）和 20 层滑动窗口注意力（sw.0-sw.19）的模型，按“分配”章节案例2分为3组：

- 第0组：10 层全局注意力（full.0 - full.9）
- 第1组：10 层滑动窗口注意力（sw.0 - sw.9）
- 第2组：10 层滑动窗口注意力（sw.10 - sw.19）

对于一次请求，会分配 11 个 block，`block_id` 0-6 属于第0组，7-8 属于第1组，9-10 属于第2组。

在此示例中，物理内存被分为 10 个 buffer（`KVCacheTensor` 0 - `KVCacheTensor` 9）。每个 buffer 被3个层共享（如 `KVCacheTensor` 0 被 full.0、sw.0、sw.10 共享），并被切分成若干块，每块大小为 `block_size * kv_hidden_size`。3 个注意力层的 KV 缓存会根据分配得到的 `block_ids` 存储到 buffer 的不同块中：

![内存布局示例](../assets/design/hybrid_kv_cache_manager/memory_layout.png)

!!! note