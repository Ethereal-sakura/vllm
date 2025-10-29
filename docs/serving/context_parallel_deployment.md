# Context 并行部署

Context 并行主要用于解决长上下文请求的服务问题。由于 prefill（预填充）和 decode（解码）阶段的特性和服务级别目标（SLO，Service Level Objectives）差异较大，因此需要分别为它们实现 context 并行。主要考虑点如下：

- 针对长上下文的 prefill，我们需要通过在查询 token 间均摊计算时间，从而控制 TTFT（首次 token 响应时间，Time to First Token）。
- 针对长上下文的 decode，我们需要更大的 KV cache 空间，从而提升 batchsize（批量大小），进而提升整体吞吐量。

## Prefill Context 并行

在 prefill 阶段，对于有 `T` 个新 token 的长请求，需要为这些新 token 计算 query/key/value 张量。假设有 `N` 个 GPU，可以将请求分为 `N` 个分片，每个 GPU 负责一部分 query/key/value 张量的计算。

根据实际需求，有两种策略：

1. 部分 query，完整 key/value：如果请求的 token 长度适中（有能力存放完整的 key/value 张量），且目标是加速 prefill（通过在 query token 间均摊计算时间），可以将所有 GPU 的 key/value 张量汇集起来，每个 GPU 仅计算自己负责的 query token 的 attention 输出。
2. 部分 query，部分 key/value：如果请求的 token 长度过长，已无法存放全部 key/value 张量，则每个 GPU 只能计算自己负责的 query/key/value 张量片段，并借助如 [ring-attention](http://arxiv.org/abs/2310.01889) 这样的技术，实现 key/value 张量的分块发送与接收。

这两种方案目前都在积极开发中。

## Decode Context 并行

由于解码阶段是自回归的，每一步都需要针对大量 KV cache 中的 key/value token 计算少量的 query token。Decode context 并行的核心在于如何将 KV cache 在多 GPU 间高效分片。

对于一个有 `H` 个 kv-head 的模型，若请求上下文有 `T` 个 token，则 KV cache 需要存储 `H * T` 个 key/value 张量。

1. 如果单个 GPU 能够全部容纳且性能满足需求，则无需并行化。
2. 如果单个 GPU 无法全部容纳，或者希望 KV cache 能支持更多请求，可以先在 `H` 维度对 KV cache 进行分片，这就是常规的 tensor parallel（张量并行）分片。只需在命令行中加上 `-tp <num_gpus>` 参数即可。
3. 由于 `H` 是由模型结构决定的并且有限，当继续增加 tensor parallel 的规模时，每个 GPU 的 KV cache 会被重复 `tp_size / H` 次。重复存储会影响效率，这时可以通过 decode context parallel 进一步在 `T` 维度分片 KV cache。只需添加 `-dcp <size>` 参数即可。注意，`size` 并不会增加实际需要启动的 GPU 数量，而是降低 KV cache 的重复度。dcp 的取值范围是 `[1, tp_size/H]`。dcp 取值越大，KV cache 重复度越低，但通信开销也会增加。

理论上，可以将 dcp 进一步扩展到超过 `tp_size / H`，以继续分片 KV cache 并加速解码。但由于解码阶段的 query token 数量有限，对于多余的 `dcp_size - tp_size / H` 个 GPU 在非 attention 层的处理方式尚不明确。为简化方案，dcp 的最大取值限制为 `tp_size / H`。如果希望进一步加速解码，可以先提升 `tp_size`，再增大 dcp。

需要注意的是，kv cache 会随着解码过程动态增长，分片策略也需要谨慎设计。我们采用交错分片策略，在 `T` 维度上对 KV cache 进行分片，这样未来新生成的 token 也能自然地在 `T` 维度分片。该方法由 [Moonshot 的 Chao Hong](https://github.com/youzhedian) 提出，并在 [这篇论文](http://arxiv.org/abs/2507.07120) 中详细阐述。

案例分析：

对于 DeepSeek-R1，启用 MLA 时只有 1 个 kv-head。常规单节点部署指定 `-tp 8` 时，KV cache 会被重复存储 8 倍。可以考虑加上 `-dcp 8`，以减少 KV cache 的重复。

对于 Kimi-K2，其结构和 DeepSeek-R1 类似，但参数更多。部署时使用 `-tp 16`，KV cache 会重复 16 倍。此时可加 `-dcp 16`，彻底消除 KV cache 的冗余，但通信开销也会增加。也可以用 `-dcp 8`，将 KV cache 重复降为 2 倍。虽然仍有两倍冗余，但通信只在单节点内，开销较小。

对于 Qwen3-235B-A22B，有 4 个 kv-head。若用 `-tp 8` 部署，KV cache 重复 2 倍。此时加上 `-dcp 2` 可以消除冗余。

简而言之，在 decode context 并行中，建议先提升 `-tp` 直至性能满足需求，再增加 `-dcp` 以减少 KV cache 重复。

vLLM 已支持 decode context parallel，适用于 MLA 和 GQA 模型。一些 attention backend 还支持将 decode context parallel 与 MTP（多 token 预测）结合，进一步加速解码。

## 技术讨论

主要讨论集中在 [vLLM Slack](https://slack.vllm.ai/) 的 `#sig-context-parallel` 频道。