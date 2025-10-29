# 自动前缀缓存

## 简介

自动前缀缓存（Automatic Prefix Caching，简称 APC）会缓存已存在查询的 KV 缓存（KV cache），这样如果新的查询和某个已存在查询拥有相同的前缀，就可以直接复用 KV 缓存，从而跳过重复部分的计算。

!!! note
    有关 vLLM 实现 APC 的技术细节，请参阅 [这里](../design/prefix_caching.md) 

## 如何在 vLLM 中启用 APC

只需在 vLLM 引擎中设置 `enable_prefix_caching=True` 即可启用 APC。示例代码如下：

[examples/offline_inference/automatic_prefix_caching.py](../../examples/offline_inference/automatic_prefix_caching.py)

## 示例场景

以下两个场景可以充分发挥 APC 的性能优势：

- 长文档查询：当用户针对同一份长文档（比如软件手册或年度报告）进行多次不同的查询时，以往每次都要重复处理整个长文档。而启用 APC 后，vLLM 只需*处理一次*长文档，后续所有请求都可以复用这个文档的 KV 缓存，避免再次计算，从而大幅提升吞吐量并降低延迟。
- 多轮对话：用户在同一次聊天会话中可能多次与应用进行交流。此时，APC 可以让 vLLM 复用历史聊天内容的处理结果，每一轮对话都不需要重新计算整个聊天历史，这样可以显著提升后续请求的响应速度和系统吞吐量。

## 限制

APC 通常不会影响 vLLM 的性能表现。但需要注意的是，APC 只能加速查询的处理（即预填充阶段），无法缩短生成新 token 的时间（即解码阶段）。因此，如果 vLLM 的大部分时间都花在生成答案上（比如答案长度很长），或者新查询与已存在查询没有相同前缀（无法复用计算），APC 并不会带来性能提升。