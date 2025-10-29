# 自动前缀缓存

前缀缓存（prefix caching）kv-cache 块是一种在大语言模型（LLM）推理中非常常见的优化方式，用于避免重复计算提示（prompt）。其核心思想很简单——我们将已处理请求的 kv-cache 块进行缓存，当有新的请求前缀和之前的请求相同时，直接复用这些缓存块。由于前缀缓存基本上是“白捡的性能提升”，且不会改变模型输出，因此已经被许多公开接口（如 OpenAI、Anthropic 等）和大多数开源 LLM 推理框架（如 SGLang）广泛采用。

虽然实现前缀缓存的方法有很多，vLLM 选择了一种基于哈希（hash-based）的方案。具体来说，我们会通过块内的 token 以及块前的前缀 token，对每一个 kv-cache 块进行哈希：

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

在上面的例子中，第一个块的 KV 缓存可以通过 token “A gentle breeze stirred” 唯一标识。第三个块则需要用块内 token “laughed in the distance” 以及前缀 token “A gentle breeze stirred the leaves as children” 来唯一标识。因此，我们可以通过 `hash(tuple[components])` 计算块哈希，其中 components 包含：

* 父块哈希值：父块的哈希值。
* 块内 tokens：当前块中的 token 元组。之所以要包含精确的 token，是为了降低哈希碰撞的可能。
* 额外哈希：其它使该块唯一的值，比如 LoRA ID、多模态输入哈希（见下方例子），以及用于多租户环境下隔离缓存的 cache salt。

!!! note "注意 1"
    我们只缓存完整的块。

!!! note "注意 2"
    上述哈希结构理论上无法完全避免碰撞。不同的前缀 token 依然可能产生相同的哈希值。**如果在多租户环境下使用，建议用 SHA256 作为哈希函数**，而不是默认的内置哈希。
    SHA256 从 vLLM v0.8.3 起支持，需通过命令行参数开启。启用后每个 token 会有大约 100-200ns 的性能损耗（50k 个 token 上下文约 6ms）。

**多模态输入下的哈希示例**  
下面以多模态输入（如图片）为例，说明前缀缓存的工作方式。假设有如下请求消息：

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

它的 prompt 形式如下：

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

可以看到，token 化后 `[IMG]` 会被一串占位 token 替换，这些占位 token 会在 prefill 阶段被图片特征向量替换。前缀缓存面临的挑战在于需要区分不同的图片和占位 token。为此我们引入前端图片处理器生成的图片哈希。例如，上述 prompt 下的各块哈希如下（假设块大小为 16，一共有 41 个占位 token）：

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 2
    Parent hash: Block 1 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 3
    Parent hash: Block 2 hash
    Token IDs: <p>, ..., <p>, 4
    Extra hash: <image hash>
```

在后续内容中，我们会先介绍 vLLM v1 实现前缀缓存的数据结构，然后讲解主要的 KV 缓存操作流程（如分配、追加、释放、淘汰），最后通过一个具体例子演示端到端的前缀缓存流程。

**缓存隔离与安全性**
为提升共享环境下的隐私保护，vLLM 支持通过可选的 per-request salt 实现前缀缓存隔离。只需在请求中添加 `cache_salt`，该值会被注入到首块哈希中，确保只有同样 salt 的请求才能复用缓存的 KV 块。这样可以防止攻击者通过观测延迟差异推测缓存内容，既提升了安全性，也不会影响性能。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

通过这种方式，只有明确约定相同 salt 的用户或请求间才能共享缓存，实现信任组内缓存复用，同时隔离其他用户。

!!! note
    引擎 V0 不支持缓存隔离。

## 数据结构

vLLM v1 的前缀缓存由 KV 缓存管理器（KV cache manager）实现。核心数据结构是“Block”数据类（简化版）：

```python
class KVCacheBlock:
    # 块 ID（不可变）
    block_id: int
    # 块哈希（块满时赋值，块被淘汰时重置）
    block_hash: BlockHash
    # 当前有多少请求在使用该块
    ref_cnt: int

    # 用于构建 free 队列的双向链表指针
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

有两个设计点值得关注：

1. 初始化 KV 缓存管理器时，我们会一次性分配所有 KVCacheBlock，组成块池（block pool）。这样可以避免频繁创建 Python 对象的开销，并且能始终追踪所有块的状态。  
2. 直接在 KVCacheBlock 内嵌入双向链表指针，可以直接构建 free 队列，这有两个好处：  
    1. 可以 O(1) 时间复杂度把队列中间的块移动到队尾。  
    2. 避免引入额外的 Python 队列（比如 `deque`），减少包装层。

因此，KV 缓存管理器初始化后会有如下几个核心组件：

![组件概览](../assets/design/prefix_caching/overview.png)

* 块池（Block Pool）：KVCacheBlock 的列表。  
* 空闲块队列（Free Block Queue）：只保存头尾块指针，便于操作。  
* 缓存块（Cache blocks）：哈希 key 到块 ID 的映射。  
* 请求块（Request blocks）：请求 ID 到已分配块 ID 的映射。

## 操作流程

### 块分配

**新请求：** 调度器为新请求分配 KV 缓存块的流程如下：

1. 调度器调用 `kv_cache_manager.get_computed_blocks()` 获取已计算的块序列。这一步会根据请求中的 prompt token 哈希查找缓存块。  
2. 调度器调用 `kv_cache_manager.allocate_slots()`，执行以下步骤：  
    1. 计算所需新块数量，如果可用块不足则直接返回。  
    2. “Touch” 已命中的缓存块。即将这些块的引用计数加一，并且如果该块之前未被其它请求占用，则从 free 队列中移除，避免被淘汰。具体可见下文例子。  
    3. 通过弹出 free 队列头部分配新块。如果头部块是已缓存块，这一步也会“淘汰”该块，其他请求无法再复用它。  
    4. 如果有分配到已满的块，会立即将其加入缓存块表，方便同 batch 内其它请求复用。

**运行中请求：** 调度器为正在运行的请求分配 KV 缓存块的流程如下：

1. 调度器调用 `kv_cache_manager.allocate_slots()`，执行以下步骤：  
    1. 计算所需新块数量，若可用块不足则直接返回。  
    2. 通过弹出 free 队列头部分配新块。如头部块已缓存，这一步会“淘汰”该块，后续请求无法再用。  
    3. 将新的 token ID 追加到已有块和新块的槽位上。如果某块已满，则加入缓存块表。

**重复块**
假设块大小为 4，你发起一个请求（Request 1），prompt 为 ABCDEF，解码长度为 3：

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

此时块 0 和块 1 已缓存。再次发起相同请求（Request 2），采用贪婪采样，生成和 Request 1 完全一致的输出：

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

可以看到，块 3 是一个新的完整块，也被缓存了。但它和块 1 是冗余的，即内容完全相同。在 v0 版本中，检测到块 3 是重复块时，会将其释放，并让 Request 2 直接引用块 1，这样 Time 1 时块表变为 `[0, 1]`。但在 vLLM v1 中，块表只能追加，不能修改，所以不会把 `[0, 3]` 改成 `[0, 1]`，因此对于哈希 key E-H 会有重复块。这些重复块会在请求释放时被清理。

### 释放

当一个请求结束时，如果相关块没有被其他请求占用（引用计数为 0），就会被释放。在本例中，我们释放请求 1 及其关联的块 2、3、4、8。可以看到，这些释放的块会按“逆序”加入 free 队列。这样设计的原因是，一个请求的最后一个块通常哈希了更多 token，被其他请求复用的概率更低，应当优先被淘汰。

![请求释放后 free 队列状态](../assets/design/prefix_caching/free.png)

### 淘汰（LRU）

当 free 队列头部（即最久未使用的块）是缓存块时，需要淘汰它，防止被其他请求复用。淘汰流程如下：

1. 从 free 队列头部弹出块，这就是要淘汰的 LRU 块。  
2. 从缓存块表中移除该块 ID。  
3. 移除该块的哈希。

## 示例流程

本例假设块大小为 4（每个块可缓存 4 个 token），KV 缓存管理器总共有 10 个块。

**时刻 1：缓存为空，有新请求到达。** 分配了 4 个块，其中 3 个块已满并被缓存，第 4 个块只填满了 3/4。

![示例时刻 1](../assets/design/prefix_caching/example-time-1.png)

**时刻 2：请求 0 使块 3 填满，并需新块继续解码。** 块 3 被缓存，块 4 被分配。

![示例时刻 2](../assets/design/prefix_caching/example-time-3.png)

**时刻 3：请求 1 到达，prompt 有 14 个 token，前 10 个与请求 0 相同。** 只有前 2 个块（8 个 token）命中缓存，第 3 块只匹配了 2/4。

![示例时刻 3](../assets/design/prefix_caching/example-time-4.png)

**时刻 4：请求 0 结束并释放。** 块 2、3、4 按逆序加入 free 队列（但块 2 和 3 仍被缓存）。块 0 和 1 因被请求 1 占用，未加入 free 队列。

![示例时刻 4](../assets/design/prefix_caching/example-time-5.png)

**时刻 5：请求 1 结束并释放。**

![示例时刻 5](../assets/design/prefix_caching/example-time-6.png)

**时刻 6：请求 2 到达，prompt 有 29 个 token，前 12 个与请求 0 相同。** 注意，虽然 free 队列块顺序为 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`，但命中缓存的块（如 0、1、2）会在分配前被 touch 并移出队列，free 队列变为 `7 - 8 - 9 - 4 - 3 - 6 - 5`。最终分配到的块为 0（缓存）、1（缓存）、2（缓存）、7、8、9、4、3（被淘汰）。

![示例时刻 6](../assets/design/prefix_caching/example-time-7.png)
