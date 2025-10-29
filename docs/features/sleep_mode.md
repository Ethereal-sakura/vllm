# 睡眠模式（Sleep Mode）

vLLM 的睡眠模式（Sleep Mode）允许你在不停止服务器或卸载 Docker 容器的情况下，临时释放模型占用的大部分 GPU 显存，包括模型权重和 KV 缓存。这在 RLHF（强化学习人类反馈）、模型训练或需要节省成本的场景下非常有用，可以在推理任务间灵活释放 GPU 资源。

主要优势：

- **释放 GPU 显存**：将模型权重转移到 CPU 内存，并清除 KV 缓存，可为其他任务释放超过 90% 的 GPU 显存。
- **快速恢复**：无需完整重载模型，即可迅速唤醒引擎并恢复推理。
- **API 接口**：可以通过 HTTP 接口或 Python API 控制模型的睡眠/唤醒状态。
- **支持分布式任务**：兼容张量并行、流水线并行等分布式计算方式。
- **精细化控制**：可选择性唤醒模型权重或 KV 缓存，避免权重更新时出现显存溢出（OOM）。

!!! note
    此功能仅支持 CUDA 平台。

## 睡眠级别

一级睡眠（level 1）会将模型权重转移到 CPU 并清除 KV 缓存，KV 缓存内容会被遗忘。一级睡眠适合让引擎“休眠”后，再次启用同一个模型。此时模型权重会备份在 CPU 内存中，请确保 CPU 内存充足以存放权重。二级睡眠（level 2）会同时清除模型权重和 KV 缓存（模型的一些缓冲区，如 rope scaling tensors，仍保留在 CPU 中）。此时模型权重和 KV 缓存内容均被遗忘。二级睡眠适合切换不同模型或更新模型时使用，例如 RLHF 权重更新场景，此时不再需要原有权重。

## 使用方法

### 离线推理

只需在初始化 `LLM` 类时传入 `enable_sleep_mode=True`，即可启用睡眠模式。

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

#### Python API

```python
# 让引擎进入一级睡眠（level=1：权重转移到 CPU，KV 缓存清除）
llm.sleep(level=1)

# 唤醒引擎（恢复权重）
llm.wake_up()
```

#### RLHF 权重更新

在 RLHF 训练过程中，vLLM 支持通过 wake_up() 的 tags 参数，有选择地只唤醒模型权重或 KV 缓存。进行权重更新时，只唤醒权重（如 llm.wake_up(tags=["weights"]）），可以避免在权重更新完成前分配 KV 缓存相关的显存，从而大幅降低显存峰值，防止大模型更新时出现 OOM 问题。

你可以用 `tags=["weights"]` 或 `tags=["kv_cache"]` 控制恢复的资源，这对于 RLHF 训练和权重更新非常实用。**注意**，在所有组件尚未完全唤醒前，`is_sleeping` 会持续返回 `true`。

```python
# 让引擎进入二级深度睡眠（level=2）
llm.sleep(level=2)
# ... 获取新权重
# 只唤醒权重，防止 OOM
llm.wake_up(tags=["weights"])
# ... 执行权重更新
# 权重更新后再唤醒 KV 缓存
llm.wake_up(tags=["kv_cache"])
```

### 在线服务

要在 vLLM 服务端启用睡眠模式，需要使用 `VLLM_SERVER_DEV_MODE=1` 环境变量，并在启动命令中添加 `--enable-sleep-mode`。

#### 开发模式下的服务端

启用 `VLLM_SERVER_DEV_MODE=1` 后，将开放开发者接口，这些接口不建议对普通用户暴露。

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

#### HTTP 接口

- `POST /sleep?level=1` — 让模型进入一级睡眠（level=1）。
- `POST /wake_up` — 唤醒模型。支持通过 `tags` 查询参数进行部分唤醒（如 `?tags=weights`）。
- `GET /is_sleeping` — 查询模型当前是否处于睡眠状态。

!!! note
    只有在设置 `VLLM_SERVER_DEV_MODE=1` 时，这些接口才会开放。
