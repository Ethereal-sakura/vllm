仅在 NVIDIA CUDA 平台下，推荐使用 [uv](https://docs.astral.sh/uv/)，这是一款非常快速的 Python 环境管理工具，用于创建和管理 Python 环境。请按照 [官方文档](https://docs.astral.sh/uv/#getting-started) 安装 `uv`。安装完成后，可以通过以下命令创建一个新的 Python 环境：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```