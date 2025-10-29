# NVIDIA TensorRT Model Optimizer

[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 是一个专为 NVIDIA GPU 推理优化模型的库。它提供了用于大型语言模型（Large Language Models, LLMs）、视觉语言模型（Vision Language Models, VLMs）和扩散模型的训练后量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization Aware Training, QAT）工具。

我们推荐使用以下命令安装该库：

```bash
pip install nvidia-modelopt
```

## 使用 PTQ 量化 HuggingFace 模型

你可以通过 TensorRT Model Optimizer 仓库中提供的示例脚本对 HuggingFace 模型进行量化。用于 LLM PTQ 的主要示例脚本通常位于 `examples/llm_ptq` 目录下。

下面是一个使用 modelopt 的 PTQ API 对模型进行量化的示例：

??? code

    ```python
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    # 从 HuggingFace 加载模型
    model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

    # 选择量化配置，例如 FP8
    config = mtq.FP8_DEFAULT_CFG

    # 定义用于校准的 forward 循环函数
    def forward_loop(model):
        for data in calib_set:
            model(data)

    # 使用 PTQ 并原地替换为量化模块
    model = mtq.quantize(model, config, forward_loop)
    ```

模型量化完成后，可以通过 export API 导出为量化后的 checkpoint：

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # 已量化的模型
        export_dir,  # 导出文件存储的目录
    )
```

量化后的 checkpoint 可以直接用 vLLM 部署。例如，下面的代码展示了如何使用 vLLM 部署 `nvidia/Llama-3.1-8B-Instruct-FP8`，它是基于 `meta-llama/Llama-3.1-8B-Instruct` 量化得到的 FP8 版本：

??? code

    ```python
    from vllm import LLM, SamplingParams

    def main():
        model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"

        # 加载 modelopt checkpoint 时需指定 quantization="modelopt"
        llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if __name__ == "__main__":
        main()
    ```