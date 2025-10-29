---
toc_depth: 4
---

# 基准测试套件

vLLM 提供了全面的基准测试工具，用于性能测试和评估：

- **[Benchmark CLI](#benchmark-cli)**：`vllm bench` 命令行工具和专用基准测试脚本，支持交互式性能测试
- **[参数扫描](#parameter-sweeps)**：自动化运行 `vllm bench`，测试多种参数配置
- **[性能基准测试](#performance-benchmarks)**：开发过程中自动化 CI 性能测试
- **[夜间基准测试](#nightly-benchmarks)**：与其它方案对比的定期基准测试

[Benchmark CLI]: #benchmark-cli

## Benchmark CLI

本节将指导你如何使用 vLLM 支持的丰富数据集运行基准测试。随着新功能和新数据集的不断推出，内容也会持续更新。

### 数据集概览

<style>
th {
  min-width: 0 !important;
}
</style>

| 数据集 | 在线 | 离线 | 数据路径 |
|---------|--------|---------|-----------|
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (图片) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>注意图片需要单独下载。例如下载 COCO 2017 年训练集图片：<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (视频) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (已弃用) | ✅ | ✅ | 本地文件：`benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (图片/视频) | 🟡 | 🚧 | `synthetic` |
| RandomForReranking | ✅ | ✅ | `synthetic` |
| Prefix Repetition | ✅ | ✅ | `synthetic` |
| HuggingFace-VisionArena | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` |
| HuggingFace-MMVU | ✅ | ✅ | `yale-nlp/MMVU` |
| HuggingFace-InstructCoder | ✅ | ✅ | `likaixin/InstructCoder` |
| HuggingFace-AIMO | ✅ | ✅ | `AI-MO/aimo-validation-aime`, `AI-MO/NuminaMath-1.5`, `AI-MO/NuminaMath-CoT` |
| HuggingFace-Other | ✅ | ✅ | `lmms-lab/LLaVA-OneVision-Data`, `Aeala/ShareGPT_Vicuna_unfiltered` |
| HuggingFace-MTBench | ✅ | ✅ | `philschmid/mt-bench` |
| HuggingFace-Blazedit | ✅ | ✅ | `vdaita/edit_5k_char`, `vdaita/edit_10k_char` |
| Spec Bench | ✅ | ✅ | `wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl` |
| Custom | ✅ | ✅ | 本地文件：`data.jsonl` |

图例说明：

- ✅ - 支持
- 🟡 - 部分支持
- 🚧 - 计划支持

!!! note
    HuggingFace 数据集的 `dataset-name` 应设置为 `hf`。
    如果是本地 `dataset-path`，请设置 `hf-name` 为其 Hugging Face ID，例如：

    ```bash
    --dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
    ```

### 示例

#### 🚀 在线基准测试

<details class="admonition abstract" markdown="1">
<summary>展开更多</summary>

首先启动你的模型服务：

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

然后运行基准测试脚本：

```bash
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

运行成功后，你会看到如下输出：

```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total Token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```

##### 自定义数据集

如果你想测试的数据集 vLLM 尚未支持，也可以通过 `CustomDataset` 方式进行基准测试。数据需为 `.jsonl` 格式，每条数据包含 "prompt" 字段，例如 data.jsonl：

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# 启动服务
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```bash
# 运行基准测试脚本
vllm bench serve --port 9001 --save-result --save-detailed \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path <path-to-your-data-jsonl> \
  --custom-skip-chat-template \
  --num-prompts 80 \
  --max-concurrency 1 \
  --temperature=0.3 \
  --top-p=0.75 \
  --result-dir "./log/"
```

如果你的数据已经包含了 chat 模板，可以加上 `--custom-skip-chat-template` 跳过自动套用。

##### 视觉语言模型 VisionArena 基准测试

```bash
# 需用支持视觉能力的模型
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

##### InstructCoder 基准测试（推理采样）

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 2048
```

##### Spec Bench 基准测试（推理采样）

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

[SpecBench 数据集](https://github.com/hemingkx/Spec-Bench)

运行所有类别：

``` bash
# 下载数据集
# wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl

vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1
```

可用类别包括 `[writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, qa, math_reasoning, rag]`。

仅运行 "summarization" 这一类：

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1
    --spec-bench-category "summarization"
```

##### 其它 HuggingFaceDataset 使用示例

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

`lmms-lab/LLaVA-OneVision-Data`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path AI-MO/aimo-validation-aime \
    --num-prompts 10 \
    --seed 42
```

`philschmid/mt-bench`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80
```

`vdaita/edit_5k_char` 或 `vdaita/edit_10k_char`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path vdaita/edit_5k_char \
    --num-prompts 90 \
    --blazedit-min-distance 0.01 \
    --blazedit-max-distance 0.99
```

##### 采样参数演示

使用 OpenAI 兼容后端（如 `vllm`）时，可以指定采样参数。示例命令如下：

```bash
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --top-k 10 \
  --top-p 0.9 \
  --temperature 0.5 \
  --num-prompts 10
```

##### 请求速率渐进（Ramp-Up）参数演示

基准测试工具还支持在测试过程中逐步提升请求速率，非常适合压力测试服务器或发现最大吞吐量。

支持两种渐进策略：

- `linear`：请求速率线性增长
- `exponential`：请求速率指数增长

可用参数如下：

- `--ramp-up-strategy`：选择渐进策略（`linear` 或 `exponential`）
- `--ramp-up-start-rps`：起始请求速率
- `--ramp-up-end-rps`：结束请求速率

##### 负载模式配置

vLLM 的基准测试脚本通过三个关键参数模拟复杂的负载模式，控制请求生成方式和并发行为：

###### 负载模式控制参数

- `--request-rate`：设置目标请求速率（每秒请求数）。`inf` 表示最大吞吐，具体数值可模拟受控负载。
- `--burstiness`：通过 Gamma 分布（取值 > 0）控制流量波动。值越低流量越突发，值越高流量越均匀。
- `--max-concurrency`：限制最大并发请求数。不设置则无限制，设定后可模拟真实的反压效果。

这三个参数可灵活组合，模拟从极限压力到生产场景下的各种流量模式。`--request-rate` 默认为 `inf`，即全部请求立刻发送，用于最大吞吐测试。若设定为有限值，默认 `--burstiness=1.0`，采用泊松过程或 Gamma 分布生成更真实的请求。`--burstiness` 只在 `--request-rate` 有限时生效，1.0 为自然泊松流量，0.1-0.5 为高突发流，2.0-5.0 为均匀流。`--max-concurrency` 默认为无限制，可设值模拟负载均衡器或 API 网关的连接上限。三者结合，可从无限压力测试 (`--request-rate=inf`) 到真实生产负载灵活模拟。

`--burstiness` 用 Gamma 分布数学控制请求到达模式：

- 形状参数：`burstiness` 值
- 变异系数（CV）：$\frac{1}{\sqrt{burstiness}}$
- 流量特性：
    - `burstiness = 0.1`：极端突发流（CV ≈ 3.16），适合压力测试
    - `burstiness = 1.0`：自然泊松流（CV = 1.0），适合基线性能
    - `burstiness = 5.0`：均匀流（CV ≈ 0.45），适合稳定测试

![负载模式示例](../assets/contributing/load-pattern-examples.png)

*图：不同用例下负载模式示例。上排为请求累计到达时间线，下排为到达间隔分布。每列对应一种用例和参数配置。*

各场景推荐负载模式：

| 用例           | Burstiness   | 请求速率    | 最大并发 | 说明                                               |
| ---            | ---          | ---         | ---      | ---                                               |
| 最大吞吐       | N/A          | 无限        | 限制     | **最常见**：模拟无限请求下网关/负载均衡器限制 |
| 真实测试       | 1.0          | 中等(5-20)  | 无限     | 自然泊松流量，基线性能测试               |
| 压力测试       | 0.1-0.5      | 高(20-100)  | 无限     | 挑战性突发流量，测试鲁棒性                 |
| 延迟分析       | 2.0-5.0      | 低(1-10)    | 无限     | 均匀流，用于时延分析                       |
| 容量规划       | 1.0          | 可变        | 限制     | 结合真实约束测试资源极限                   |
| SLA 校验       | 1.0          | 目标速率    | SLA 上限 | 模拟生产约束，合规性检测                   |

这些负载模式可帮助你从基础性能到极端压力下全面评估 vLLM 部署表现。

**最大吞吐模式**（`--request-rate=inf --max-concurrency=<limit>`）是生产基准测试最常用配置。它模拟了实际部署结构：

- 用户端以最大速率发送请求
- 负载均衡器/网关控制最大并发数
- 系统以最大并发运行，挖掘真实吞吐极限
- 此时 `--burstiness` 参数无效

此模式有助于确定生产环境下最优并发配置。

为合理配置负载参数，尤其用于**容量规划**和**SLA 校验**，需了解系统资源极限。vLLM 启动时会报告 KV cache 配置，直接影响负载参数：

```text
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
``