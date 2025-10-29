# AutoAWQ

> ⚠️ **注意：**  
    `AutoAWQ` 库已不再维护。该功能现已被 vLLM 项目在 [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) 中吸收。  
    推荐的量化流程请参考 [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) 中的 AWQ 示例。关于停用的更多细节，请参见原始 [AutoAWQ 仓库](https://github.com/casper-hansen/AutoAWQ)。

如果你想创建一个新的 4-bit 量化模型，可以使用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。  
量化会将模型的精度从 BF16/FP16 降低到 INT4，从而有效减少模型整体的内存占用。  
这样做的主要好处是延迟更低、内存消耗更小。

你可以通过安装 AutoAWQ 量化你自己的模型，或者直接选用 Huggingface 上已有的 [6500+ AWQ 量化模型](https://huggingface.co/models?search=awq)。

```bash
pip install autoawq
```

安装好 AutoAWQ 之后，就可以开始对模型进行量化了。更多细节可以参考 [AutoAWQ 官方文档](https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization)。下面是如何量化 `mistralai/Mistral-7B-Instruct-v0.2` 的一个示例：

??? code

    ```python
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    quant_path = "mistral-instruct-v0.2-awq"
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # 加载模型
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 进行量化
    model.quantize(tokenizer, quant_config=quant_config)

    # 保存量化后的模型
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'模型已量化并保存在 "{quant_path}"')
    ```

如果你想用 vLLM 运行 AWQ 量化模型，可以用如下命令加载 [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ)：

```bash
python examples/offline_inference/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq
```

AWQ 量化模型也可以直接通过 LLM 接口使用：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 示例提示
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # 创建采样参数对象
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 创建 LLM 实例
    llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
    # 根据提示生成文本。输出为 RequestOutput 对象列表，
    # 包含原始 prompt、生成的文本以及其他信息。
    outputs = llm.generate(prompts, sampling_params)
    # 打印输出内容
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```