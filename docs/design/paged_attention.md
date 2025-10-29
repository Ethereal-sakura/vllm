# 分页注意力机制（Paged Attention）

!!! warning
    本文档是基于 [vLLM 原始论文](https://arxiv.org/abs/2309.06180) 的历史内容。
    文中描述的内容已不再适用于当前 vLLM 的代码实现。

目前，vLLM 使用了自己的多头查询注意力核函数（multi-head query attention kernel），实现代码在 `csrc/attention/attention_kernels.cu`。
该核函数专门为 vLLM 的分页 KV 缓存（paged KV caches）设计，其中 key 和 value 缓存分别存储在独立的块中（注意，这里的块和 GPU 的线程块（thread block）不同。所以在后文中，我将 vLLM 的分页注意力块称为“块”，而 GPU 的线程块称为“线程块”）。

为了实现高性能，这个核函数依赖于特别设计的内存布局和访问方式，尤其是在每个线程从全局内存读取数据到共享内存的环节。本文档将一步步为大家解释该核函数的整体实现逻辑，帮助想要深入学习 vLLM 多头查询注意力机制的读者。阅读完本篇内容后，你会对其实现流程有更清晰的理解，也能更容易跟进源码细节。

请注意，本文档不会涵盖所有细节，比如如何计算对应数据的正确索引、点乘实现等。但在掌握了高层逻辑之后，理解实际代码会变得更加轻松。

## 输入参数

核函数会接收一系列参数，让当前线程能够完成分配到的工作。最重要的三个参数是输入指针 `q`、`k_cache` 和 `v_cache`，分别指向全局内存中的查询（query）、键（key）、值（value）数据，这些数据需要被读取和处理。输出指针 `out` 指向全局内存，用于存储最终结果。这四个指针实际都代表多维数组，但每个线程只会访问自己负责的数据部分。其他运行时参数这里省略，仅做简化。

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // 其他参数
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // 其他参数
)
```

函数签名上方的模板参数会在编译时确定。`scalar_t` 表示查询、键和值数据的元素类型，比如 FP16。`HEAD_SIZE` 是每个头的元素数量。`BLOCK_SIZE` 表示每个块内的 token 数量。`NUM_THREADS` 是每个线程块中的线程数。`PARTITION_SIZE` 代表张量并行的 GPU 数量（为简化起见，这里我们假设为 0，即不启用张量并行）。

有了这些参数后，需要进行一系列准备工作，比如计算当前头索引、块索引等其它变量。不过这些准备步骤可以先不考虑，等整体流程理解清楚后再回头看会更容易。

## 重要概念

在正式介绍计算流程前，先补充一些后续章节需要用到的重要术语。如果暂时不明白，后面遇到相关名词时可以回来看。

- **序列（Sequence）**：一个序列代表一次客户端请求。比如 `q` 指向的数据形状为 `[num_seqs, num_heads, head_size]`，说明共有 `num_seqs` 个查询序列。由于该核函数是单查询注意力机制，每个序列仅有一个查询 token，因此 `num_seqs` 就是 batch 中处理的 token 总数。
- **上下文（Context）**：上下文由序列生成的所有 token 组成。例如，`["What", "is", "your"]` 是上下文 token，输入查询 token 为 `"name"`，模型可能生成 `"?"`。
- **向量（Vec）**：vec 是一组共同读取和计算的元素。对于查询和键数据，vec 的大小（`VEC_SIZE`）按每个线程组能一次读取并计算 16 字节数据来确定。对于值数据，vec 大小（`V_VEC_SIZE`）则按每个线程一次能读取并计算 16 字节来设置。例如，`scalar_t` 若为 FP16（2 字节），`THREAD_GROUP_SIZE` 为 2，则 `VEC_SIZE=4`，`V_VEC_SIZE=8`。
- **线程组（Thread group）**：线程组是一小组线程（`THREAD_GROUP_SIZE`），每次负责读取和计算一个查询 token 和一个键 token。每个线程只处理该 token 数据的一部分。线程组处理的元素总数称为 `x`。比如线程组有 2 个线程，头大小为 8，则线程 0 处理索引 0、2、4、6 的元素，线程 1 处理索引 1、3、5、7。
- **块（Block）**：vLLM 的键和值缓存被分割为若干块。每个块存储某个头下固定数量（`BLOCK_SIZE`）的 token 数据。每块可能只包含上下文 tokens 的一部分。例如块大小为 16、头大小为 128，则单个块可存储 16 * 128 = 2048 个元素。
- **Warp**：一个 warp 是 32 个线程（`WARP_SIZE`）的集合，在同一个流多处理器（SM）上并行执行。在该核函数中，每个 warp 每次处理一个查询 token 和一个块中的所有键 token 的计算（多次迭代可处理多个块）。比如有 4 个 warp、一个上下文有 6 个块，warp 0 处理第 0、4 块，warp 1 处理第 1、5 块，warp 2 处理第 2 块，warp 3 处理第 3 块。
- **线程块（Thread block）**：线程块是一组线程（`NUM_THREADS`），可访问同一共享内存。每个线程块包含多个 warp（`NUM_WARPS`），本核函数中每个线程块负责一个查询 token 与整个上下文内所有键 token 的计算。
- **网格（Grid）**：网格是所有线程块的集合，定义了整体的形状。在本核函数中，形状为 `(num_heads, num_seqs, max_num_partitions)`。因此每个线程块只处理一个头、一个序列和一个分区的数据计算。

## 查询（Query）

本节介绍查询数据在内存中的存储方式以及每个线程如何读取。前文提到，每个线程组会读取一个查询 token 的数据，而每个线程只处理该 token 的一部分。在同一个 warp 内，每个线程组都会读取同一个查询 token 的数据，但会与不同的键 token 数据做点乘。

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

<figure markdown="span">
  ![](../assets/design/paged_attention/query.png){ align="center" alt="query" width="70%" }
</figure>

每个线程会定义自己的 `q_ptr`，指向全局内存中分配到的查询 token 数据。例如 `VEC_SIZE=4`、`HEAD_SIZE=128` 时，`q_ptr` 指向的这段数据包含 128 个元素，被分为 128 / 4 = 32 个 vec。

<figure markdown="span">
  ![](../assets/design/paged_attention/q_vecs.png){ align="center" alt="q_vecs" width="70%" }
</figure>

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

接下来，需将 `q_ptr` 指向的全局内存数据读取到共享内存 `q_vecs`。要注意的是，每个 vec 分配到不同行。例如，`THREAD_GROUP_SIZE=2` 时，线程 0 负责第 0 行 vec，线程 1 负责第 1 行 vec。这样读取查询数据可以让相邻线程（如线程 0 和线程 1）访问相邻内存，实现内存合并（memory coalescing），提升性能。

## 键（Key）

与“查询”类似，本节介绍键数据的内存布局及分配方式。每个线程组每次只处理一个查询 token，但可以在多次迭代中处理多个键 token。每个 warp 会通过多次迭代处理多个块，确保所有上下文 token 都被处理一次。在这里，“处理”指的是查询与键数据的点乘计算。

```cpp
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

与 `q_ptr` 不同，每个线程的 `k_ptr` 会在不同迭代中指向不同的键 token。上述代码说明，`k_ptr` 会根据分配的块、头、token 在 `k_cache` 上定位键 token 数据。

<figure markdown="span">
  ![](../assets/design/paged_attention/key.png){ align="center" alt="key" width="70%" }
</figure>

上图展示了键数据的内存布局。假设 `BLOCK_SIZE=16`、`HEAD_SIZE=128`、`x=8`、`THREAD_GROUP_SIZE=2`，且共有 4 个 warp。每个矩形代表一个头下某个键 token 的全部元素，由一个线程组处理。左半部分展示了 warp 0 负责的 16 块键 token 数据，右半部分是其它 warp 或迭代处理的剩余数据。每个矩形内有 32 个 vec（每 token 128 个元素），由 2 个线程分别处理。

<figure markdown="span">
  ![](../assets/design/paged_attention/k_vecs.png){ align="center" alt="k_vecs" width="70%" }
</figure>

```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD]
```

接下来，需要把 `k_ptr` 指向的键 token 数据读取到寄存器 `k_vecs`。这里使用寄存器是因为 `k_vecs` 只会被单个线程访问一次，而 `q_vecs` 可能被多个线程反复访问。每个 `k_vecs` 会存储多个后续计算用的向量，每次内层迭代设置一个 vec。vec 的分配方式可以让 warp 中相邻线程一起读取相邻内存，同样实现内存合并。例如线程 0 读取 vec 0，线程 1 读取 vec 1，下一次内层循环分别读取 vec 2、vec 3，以此类推。

如果对整体流程还不太清楚，不必担心，继续阅读下方 “QK” 部分，会以更高层、更直观的方式说明查询与键的计算流程。

## QK

如下伪代码所示，在整个外层循环前，先读取一个查询 token 的数据到 `q_vecs`。然后在外层循环中，遍历不同的 `k_ptr`，分别指向不同的键 token，并在内层循环准备 `k_vecs`。最后，将 `q_vecs` 与 `k_vecs` 做点乘。

```cpp
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

如前所述，每个线程每次只读取部分查询和键 token 数据。不过在 `Qk_dot<>::dot` 中会跨线程组做归约，因此这里返回的 `qk` 并不是部分数据的点乘结果，而是整个查询与键 token 的完整点乘结果。

举例来说，`HEAD_SIZE=128`、`THREAD_GROUP_SIZE=2` 时，每个线程的 `k_vecs` 包含 64 个元素。但最终返回的 `qk` 是 128 个查询元素与 128 个键元素的点乘结果。如果想了解具体点乘和归约的细节，可以参考 `Qk_dot<>::dot` 的实现，本文不再展开。

## Softmax

接下来需对所有 `qk` 计算归一化的 softmax，如上图所示，每个 $x$ 代表一个 `qk`。这一步需要获得所有 `qk` 的最大值 `qk_max`（$m(x)$）和指数和 `exp_sum`（$\ell(x)$）。归约操作需要在整个线程块范围内完成，包含查询 token 与所有上下文键 token 的结果。

$$
\begin{gather*}
m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\
\quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{gather*}
$$

### `qk_max` 和 `logits`

刚拿到 `qk` 结果后，可以把临时的 `logits` 设置为 `qk`（最终 `logits` 会存储归一化 softmax 结果）。同时可以比较并收集当前线程组计算出的所有 `qk` 的最大值 `qk_max`。

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= context_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

请注意，`logits` 存储在共享内存，每个线程组会为分配到的上下文 token 设置对应字段。整体而言，logits 的大小等于上下文 token 数量。

```cpp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
```

随后，要在每个 warp 内归约 `qk_max`。主要通过 warp 内线程互相通信，得到最终的最大 `qk`。

```cpp
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

最后，将每个线程块所有 warp 的 `qk_max` 进行归约，得到整个线程块的最大值，并广播给每个线程。

### `exp_sum`

同样地，exp_sum 也需要在整个线程块范围内归约。

```cpp
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
...
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

首先，对每个线程组的 exp 值求和，同时将 `logits` 从 `qk` 转化为 `exp(qk - qk_max)`。注意，这里的 `qk_max` 已是整个线程块的最大值。之后，对 `exp_sum` 做线程块级归约，和 `qk_max` 的归约方式相同。

```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
```

最终，使用归约得到的 `qk_max` 和 `exp_sum`，可以得到归一化 softmax 结果并存入 `logits`。后续会用该变量与值数据做点乘。此时，`logits` 已存储了所有分配到的上下文 token 的归一化 `qk` softmax 结果。

## 值（Value）

<figure markdown="span">
  ![](../assets/design/paged_attention/value.png){ align="center" alt="value" width="70%" }
</figure>

<figure markdown="span">
  ![](../assets/design/paged_attention/logits_vec.png){ align="center" alt="logits_vec" width="50%" }
</figure>

<figure markdown="span">
  ![](../assets/design/paged_attention/v_vec.png){ align="center" alt="v_vec" width="70%" }
</figure>

现在需要取出值数据，与 logits 做点乘。不同于查询和键，值数据没有线程组的概念。上图所示，值 token 的内存布局和键 token 不同，同一列的元素属于同一个值 token。对于单个值块（block），有 `HEAD_SIZE` 行和 `BLOCK_SIZE` 列，被分割成多个 `v_vecs`。

每个线程每次都从同一组 token 中读取 `V_VEC_SIZE` 个元素。因此