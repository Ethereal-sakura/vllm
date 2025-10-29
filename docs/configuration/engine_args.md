---
toc_depth: 3
---

# 引擎参数

引擎参数用于控制 vLLM 引擎的行为。

- 在[离线推理](../serving/offline_inference.md)场景下，它们是传递给 [LLM][vllm.LLM] 类的参数之一。
- 在[在线服务](../serving/openai_compatible_server.md)场景下，它们作为 `vllm serve` 命令的参数。

引擎参数类，[EngineArgs][vllm.engine.arg_utils.EngineArgs] 和 [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs]，实际上是 [vllm.config][] 中定义的多个配置类的组合。因此，如果你需要开发相关的文档，建议直接查阅这些配置类，因为它们是类型、默认值和文档注释的权威来源。

--8<-- "docs/cli/json_tip.inc.md"

## `EngineArgs`

--8<-- "docs/argparse/engine_args.md"

## `AsyncEngineArgs`

--8<-- "docs/argparse/async_engine_args.md"
