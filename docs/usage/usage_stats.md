# 使用统计数据收集

vLLM 默认会收集匿名使用数据，帮助工程团队更好地了解哪些硬件和模型配置被广泛采用。这些数据可以让团队将开发重点放在最常见的工作负载上。收集的数据完全透明，并且不会包含任何敏感信息。

经过清洗和汇总后，部分数据会公开发布，方便社区参考。例如，你可以在[这里](https://2024.vllm.ai)查看 2024 年的使用报告。

## 收集哪些数据？

vLLM 最新版本收集的数据列表可以在这里找到：[vllm/usage/usage_lib.py](../../vllm/usage/usage_lib.py)

以下是 v0.4.0 版本的数据示例：

??? console "输出示例"

    ```json
    {
      "uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
      "provider": "GCP",
      "num_cpu": 24,
      "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
      "cpu_family_model_stepping": "6,85,7",
      "total_memory": 101261135872,
      "architecture": "x86_64",
      "platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
      "gpu_count": 2,
      "gpu_type": "NVIDIA L4",
      "gpu_memory_per_device": 23580639232,
      "model_architecture": "OPTForCausalLM",
      "vllm_version": "0.3.2+cu123",
      "context": "LLM_CLASS",
      "log_time": 1711663373492490000,
      "source": "production",
      "dtype": "torch.float16",
      "tensor_parallel_size": 1,
      "block_size": 16,
      "gpu_memory_utilization": 0.9,
      "quantization": null,
      "kv_cache_dtype": "auto",
      "enable_lora": false,
      "enable_prefix_caching": false,
      "enforce_eager": false,
      "disable_custom_all_reduce": true
    }
    ```

你可以通过以下命令预览已收集的数据：

```bash
tail ~/.config/vllm/usage_stats.json
```

## 如何关闭数据收集

你可以通过设置环境变量 `VLLM_NO_USAGE_STATS` 或 `DO_NOT_TRACK`，或者在 `~/.config/vllm` 目录下创建 `do_not_track` 文件来关闭使用统计数据收集：

```bash
# 以下任意一种方式都可以关闭使用统计数据收集
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```