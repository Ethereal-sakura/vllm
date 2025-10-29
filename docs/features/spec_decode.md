# 推测解码（Speculative Decoding）

!!! warning
    请注意，目前 vLLM 中的推测解码功能尚未经过优化，对于所有提示数据集或采样参数，并不能保证带来显著的跨 token 延迟下降。相关优化工作正在持续进行，可在这里跟进进展：<https://github.com/vllm-project/vllm/issues/4630>

!!! warning
    当前 vLLM 的推测解码功能暂不支持流水线并行（pipeline parallelism）。

本文介绍如何在 vLLM 中使用[推测解码（Speculative Decoding）](https://x.com/karpathy/status/1697318534555336961)。推测解码是一种能够提升大语言模型（LLM）推理跨 token 延迟的技术，适用于内存瓶颈场景。

## 使用草稿模型进行推测解码

以下代码展示了如何在 vLLM 的离线模式下，结合草稿模型进行推测解码，每次推测 5 个 token。

!!! warning
    在 vllm v0.10.0 版本中，暂不支持基于草稿模型的推测解码。
    如果使用下述代码，将会遇到 `NotImplementedError` 错误。

??? code

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        speculative_config={
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 5,
        },
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

如果想要在在线模式下实现相同功能，可以先启动服务端：

```bash
vllm serve facebook/opt-6.7b \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 42 \
    -tp 1 \
    --gpu_memory_utilization 0.8 \
    --speculative_config '{"model": "facebook/opt-125m", "num_speculative_tokens": 5}'
```

!!! warning
    注意：请统一使用 `--speculative_config` 参数来设置所有与推测解码相关的配置。之前通过 `--speculative_model` 指定模型，并单独添加相关参数（如 `--num_speculative_tokens`）的方式已被废弃。

然后可以使用客户端进行调用：

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和 API base，指向 vLLM 的 API 服务。
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # 默认会读取 os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Completion API
    stream = False
    completion = client.completions.create(
        model=model,
        prompt="The future of AI is",
        echo=False,
        n=1,
        stream=stream,
    )

    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    ```

## 基于提示 n-gram 匹配进行推测解码

下面的代码配置 vLLM 使用基于提示 n-gram 匹配生成推测 token 的方法进行推测解码。更多信息可参考[这个讨论串](https://x.com/joao_gante/status/1747322413006643259)。

??? code

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        },
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

## 使用 MLP 推测器（speculator）进行推测解码

下面的代码配置 vLLM，采用草稿模型，结合上下文向量和采样 token 来生成推测内容，实现推测解码。详细介绍可参考[这篇博客](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)或[这份技术报告](https://arxiv.org/abs/2404.19124)。

??? code

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tensor_parallel_size=4,
        speculative_config={
            "model": "ibm-ai-platform/llama3-70b-accelerator",
            "draft_tensor_parallel_size": 1,
        },
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

需要注意的是，这类推测模型当前必须在 draft 阶段关闭张量并行（tensor parallelism），不过主模型可以开启张量并行（如上例所示）。由于推测模型体积较小，依然能够带来明显的加速效果。未来版本会修复该限制。

目前 Hugging Face hub 上已提供多种此类型的推测模型：

- [llama-13b-accelerator](https://huggingface.co/ibm-ai-platform/llama-13b-accelerator)
- [llama3-8b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-8b-accelerator)
- [codellama-34b-accelerator](https://huggingface.co/ibm-ai-platform/codellama-34b-accelerator)
- [llama2-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama2-70b-accelerator)
- [llama3-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-70b-accelerator)
- [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)
- [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)
- [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)
- [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)

## 基于 EAGLE 草稿模型进行推测解码

下面的代码配置 vLLM，使用基于 [EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）](https://arxiv.org/pdf/2401.15077) 的草稿模型进行推测解码。更详细的离线模式示例（包括如何提取请求级别的接受率）可参考[这里](../../examples/offline_inference/spec_decode.py)。

??? code

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=4,
        speculative_config={
            "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 2,
            "method": "eagle",
        },
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    ```

使用 EAGLE 草稿模型时，需要注意以下几点：

1. [HF EAGLE 模型仓库](https://huggingface.co/yuhuili) 中的 EAGLE 草稿模型，在 <https://github.com/vllm-project/vllm/pull/12304> 合入后，可以被 vLLM 直接加载使用。  
   如果你使用的是早于该 PR 的 vllm 版本，请用[这个脚本](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)转换推测模型，并在 `speculative_config` 里指定 `"model": "path/to/modified/eagle/model"`。如果在最新版 vLLM 下加载权重仍有问题，欢迎在仓库留言或提 issue。

2. EAGLE 草稿模型 draft 阶段必须关闭张量并行（即 `speculative_config` 中的 draft_tensor_parallel_size 需为 1），但主模型可以开启并行（如上例所示）。

3. 在 vLLM 中使用 EAGLE 推测器时，整体加速效果相比[官方实现](https://github.com/SafeAILab/EAGLE)会低一些。该问题正在调查中，进展可跟踪：<https://github.com/vllm-project/vllm/issues/9565>

4. 针对 EAGLE-3 草稿模型，"method" 选项必须设置为 "eagle3"，也就是在 `speculative_config` 内指定 `"method": "eagle3"`。

Hugging Face hub 上已发布多种 EAGLE 草稿模型：

| 基础模型                                                              | EAGLE 模型仓库                           | EAGLE 参数量       |
|---------------------------------------------------------------------|-------------------------------------------|--------------------|
| Vicuna-7B-v1.3                                                       | yuhuili/EAGLE-Vicuna-7B-v1.3             | 0.24B              |
| Vicuna-13B-v1.3                                                      | yuhuili/EAGLE-Vicuna-13B-v1.3            | 0.37B              |
| Vicuna-33B-v1.3                                                      | yuhuili/EAGLE-Vicuna-33B-v1.3            | 0.56B              |
| LLaMA2-Chat 7B                                                       | yuhuili/EAGLE-llama2-chat-7B             | 0.24B              |
| LLaMA2-Chat 13B                                                      | yuhuili/EAGLE-llama2-chat-13B            | 0.37B              |
| LLaMA2-Chat 70B                                                      | yuhuili/EAGLE-llama2-chat-70B            | 0.99B              |
| Mixtral-8x7B-Instruct-v0.1                                           | yuhuili/EAGLE-mixtral-instruct-8x7B      | 0.28B              |
| LLaMA3-Instruct 8B                                                   | yuhuili/EAGLE-LLaMA3-Instruct-8B         | 0.25B              |
| LLaMA3-Instruct 70B                                                  | yuhuili/EAGLE-LLaMA3-Instruct-70B        | 0.99B              |
| Qwen2-7B-Instruct                                                    | yuhuili/EAGLE-Qwen2-7B-Instruct          | 0.26B              |
| Qwen2-72B-Instruct                                                   | yuhuili/EAGLE-Qwen2-72B-Instruct         | 1.05B              |

## 推测解码的无损性保障

在 vLLM 中，推测解码的目标是在保证推理准确性的前提下提升效率。本节将分三个方面说明推测解码的无损性保障：

1. **理论无损性**
   \- 推测解码采样在理论上是无损的，仅受限于硬件数值精度。浮点误差可能导致输出分布出现微小差异，详见
   [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

2. **算法无损性**
   \- vLLM 实现的推测解码经过算法验证，确保无损。关键验证包括：

    > - **拒绝采样器收敛性**：确保 vLLM 的拒绝采样器采样结果与目标分布一致。[查看测试代码](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **贪心采样一致性**：确保推测解码下的贪心采样与普通贪心采样结果一致。即 vLLM 推测解码框架与 vLLM 前向推理、拒绝采样器结合时，能够保证无损性。绝大多数 [tests/spec_decode/e2e](../../tests/spec_decode/e2e) 里的测试均通过[此断言实现](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291) 进行校验。

3. **vLLM logprob 稳定性**
   \- vLLM 当前不保证输出 token 的 log probability（logprob）稳定性。这意味着同一请求多次运行可能得到不同输出。详情可参阅 FAQ 中 *Can the output of a prompt vary across runs in vLLM?*，见[常见问题](../usage/faq.md)。

尽管 vLLM 力求推测解码的无损性，但在实际生成过程中，开启与关闭推测解码时输出结果可能出现差异，原因包括：

- **浮点精度问题**：硬件数值精度的差异可能导致输出分布存在细微变化
- **批量大小与数值稳定性**：批量大小变化可能引起 logprob 和输出概率的波动，通常与批量操作的非确定性和数值不稳定有关

缓解方法可参考 FAQ 中 *Can the output of a prompt vary across runs in vLLM?*，见[常见问题](../usage/faq.md)。

## vLLM 贡献者参考资料

- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
