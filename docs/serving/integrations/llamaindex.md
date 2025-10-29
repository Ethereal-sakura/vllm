# LlamaIndex

你也可以通过 [LlamaIndex](https://github.com/run-llama/llama_index) 使用 vLLM 

要安装 LlamaIndex，请运行

```bash
pip install llama-index-llms-vllm -q
```

如果你想在单块或多块 GPU 上进行推理，可以使用 `llamaindex` 中的 `Vllm` 类。

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)
```

更多详细内容请参考这个 [教程](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/) 
