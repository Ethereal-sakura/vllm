# 参与 vLLM 项目贡献

感谢你有兴趣为 vLLM 做贡献！我们的社区对所有人开放，无论贡献大小都非常欢迎。你可以通过以下多种方式参与到项目中来：

- 发现并报告任何问题或 bug
- 请求或添加对新模型的支持
- 提出或实现新特性
- 完善文档或撰写使用指南

我们也非常重视社区支持，因此，解答问题、参与 PR 审核、协助他人同样是非常宝贵且受欢迎的贡献。

最后，帮助我们提升 vLLM 的知名度也是极具影响力的支持方式。你可以在博客中介绍 vLLM，展示它如何助力你的项目。如果你正在使用 vLLM，不妨在社交媒体上表达你的支持，或者给我们的仓库点个 Star！

## 任务看板

不知道从哪里开始？可以通过以下链接寻找适合参与的任务：

- [适合新手的问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
    - [精选入门任务](https://github.com/orgs/vllm-project/projects/6)
- [新模型请求](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new-model%22)
    - [支持多模态能力的模型](https://github.com/orgs/vllm-project/projects/10)

## 许可证

请参阅 [LICENSE](../../LICENSE)。

## 开发流程

参与 vLLM 代码贡献的第一步是克隆 GitHub 仓库：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

接下来，请配置你的 Python 虚拟环境。

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

如果你只打算开发 vLLM 的 Python 代码，可以通过以下方式安装 vLLM：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

如果你需要同时开发 vLLM 的 Python 和 CUDA/C++ 代码，则使用：

```bash
uv pip install -e .
```

关于源码安装及其他硬件环境的安装细节，请参考[安装指南](../getting_started/installation/README.md)中的“从源码构建 wheel”部分。

如果你需要高效迭代 C++/CUDA kernel，建议阅读 [增量编译工作流](./incremental_build.md)。

!!! tip
    vLLM 支持 Python 3.10 到 3.13 版本。不过，vLLM 默认的 [Dockerfile](../../docker/Dockerfile) 使用 Python 3.12，并且 CI 测试（除了 `mypy`）也使用 Python 3.12。

    因此，建议你在本地开发时也采用 Python 3.12，以减少本地环境与 CI 环境不一致带来的问题。

### 代码风格检查

vLLM 使用 `pre-commit` 进行代码格式化和风格检查。如果你还不熟悉 `pre-commit`，可以访问 <https://pre-commit.com/#usage>。配置方式非常简单：

```bash
uv pip install pre-commit
pre-commit install
```

现在，每次提交代码时，vLLM 的 `pre-commit` 钩子都会自动运行。

!!! tip "小贴士"
    你可以手动运行 `pre-commit` 钩子：

    ```bash
    pre-commit run     # 只检查暂存区文件
    pre-commit run -a  # 检查所有文件（等价于 --all-files）
    ```

    ---

    部分 `pre-commit` 钩子只会在 CI 中运行。如果需要在本地执行，可以使用：

    ```bash
    pre-commit run --hook-stage manual markdownlint
    pre-commit run --hook-stage manual mypy-3.10
    ```

### 文档编写

MkDocs 是一个快速、简单、美观的静态站点生成器，非常适合项目文档编写。文档源文件采用 Markdown 编写，使用一个 YAML 配置文件 [mkdocs.yaml](../../mkdocs.yaml) 进行配置。

首先安装相关依赖：

```bash
uv pip install -r requirements/docs.txt
```

!!! tip
    请确保你的 Python 版本兼容所有插件
    （例如，`mkdocs-awesome-nav` 需要 Python 3.10 及以上）

MkDocs 自带开发服务器，可以实时预览文档。在仓库根目录下运行：

```bash
mkdocs serve                           # 启动 API 文档（首次约需 10 分钟）
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # 不生成 API 文档（约 15 秒）
```

当日志中出现 `Serving on http://127.0.0.1:8000/` 后，即可在浏览器中打开 <http://127.0.0.1:8000/> 预览文档。

更多功能和高级配置请参考：

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs 主题文档](https://squidfunk.github.io/mkdocs-material/)

### 测试

vLLM 使用 `pytest` 进行代码测试。

```bash
# 安装 CI 所用测试依赖（仅限 CUDA 环境）
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# 安装通用测试依赖（与硬件无关）
uv pip install pytest pytest-asyncio

# 运行所有测试
pytest tests/

# 仅运行某个测试文件并输出详细信息
pytest -s -v tests/test_logger.py
```

!!! tip "如果缺少 Python.h，请安装 python3-dev"
    如果上述命令报错 `Python.h: No such file or directory`，
    请通过 `sudo apt install python3-dev` 安装所需依赖。

!!! warning "注意"
    当前仓库尚未完全通过 `mypy` 类型检查。

    ---

    目前并非所有单元测试都能在 CPU 平台下通过。如果你无法访问 GPU 平台进行本地测试，可以依赖持续集成系统完成测试。

## 问题反馈

如果你遇到 bug 或有新的功能需求，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)以避免重复。如果没有相关问题，请[新建 issue](https://github.com/vllm-project/vllm/issues/new/choose)，尽可能提供详细信息。

!!! important
    如果你发现了安全漏洞，请按照[这里的指引](../../SECURITY.md)操作。

## Pull Request 与代码审核

感谢你为 vLLM 做出贡献！在提交 Pull Request 之前，请确保你的 PR 满足以下标准，这有助于我们保持代码质量并提升审核效率。

### DCO 与签名提交

参与本项目贡献时，你需要同意 [DCO](../../DCO)。所有提交必须包含 `Signed-off-by:` 头部，用于确认你同意 DCO。

使用 `git commit -s` 可自动添加该头部。

!!! tip
    你可以在 IDE 中开启自动签名提交：

    - **PyCharm**：在 `Commit` 窗口点击 `Commit and Push...` 右侧的 `Show Commit Options` 图标，在弹出的 `git` 窗口中修改 `Author` 并勾选 `Sign-off commit`。
    - **VSCode**：打开[设置编辑器](https://code.visualstudio.com/docs/configure/settings)，启用 `Git: Always Sign Off` (`git.alwaysSignOff`) 选项。

### PR 标题与分类

只有特定类型的 PR 会被审核。请在 PR 标题前添加对应前缀，以标识你的更改类型，请选择以下之一：

- `[Bugfix]` 修复 bug
- `[CI/Build]` 构建或持续集成相关改进
- `[Doc]` 文档修订与完善
- `[Model]` 新模型添加或现有模型改进，标题中应包含模型名
- `[Frontend]` vLLM 前端相关改动（如 OpenAI API server、`LLM` 类等）
- `[Kernel]` CUDA kernel 或其他计算 kernel 相关改动
- `[Core]` vLLM 核心逻辑相关改动（如 `LLMEngine`、`AsyncLLMEngine`、`Scheduler` 等）
- `[Hardware][Vendor]` 硬件相关改动，前缀中需包含厂商名（如 `[Hardware][AMD]`）
- `[Misc]` 其他无法归类的更改，请尽量少用

!!! note
    如果 PR 涉及多个类别，请在标题中包含所有相关前缀。

### 代码质量

PR 需要满足以下代码质量要求：

- 遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html) 和 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)
- 通过所有代码检查工具
- 代码需有良好注释，便于后续贡献者理解
- 包含充分的测试，保证功能正确性和鲁棒性，包括单元测试和集成测试
- 如果 PR 修改了 vLLM 对外行为，请补充或更新 `docs/` 下的相关文档，帮助用户理解和使用新功能或更改

### Kernel 添加与修改

在开发或修改 kernel 时，推荐采用 [增量编译工作流](./incremental_build.md) 以加快构建速度。每个自定义 kernel 需要有 schema，并注册一个或多个实现到 PyTorch。

- 请按照 PyTorch 指南注册自定义操作符：
  [自定义 C++ 和 CUDA 运算符](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
  和 [自定义运算符手册](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU)
- 返回 `Tensors` 的自定义操作需要实现 meta-function
  meta-function 应在 Python 中实现和注册，以便自动处理动态维度，详见上述文档
- 使用 [torch.library.opcheck()](https://pytorch.org/docs/stable/library.html#torch.library.opcheck) 检查自定义操作符和 meta-function 的注册，参考 `tests/kernels`
- 如果修改了现有操作符的 C++ 签名，记得同步更新其 schema
- 如需新增自定义类型，请参考：
  [PT2 的自定义类支持](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA)

### 大型变更的注意事项

请尽量精简你的变更内容。如涉及重大架构调整（代码行数超过 500，不含 kernel/data/config/test），请先在 GitHub issue（RFC）中讨论技术设计和理由，否则我们会标记为 `rfc-required` 并可能拒绝合并。

### 审核流程说明

我们的目标是成为一个 *透明高效的审核团队*。我们希望流程公开透明，让每位贡献者都能清楚进展、不感到疑惑或挫败。但由于团队人力有限，我们需对部分 PR 进行优先级排序。审核流程如下：

- PR 提交后，将分配给一位审核者。每位审核者会根据自己的专长和时间安排领取 PR
- PR 分配后，审核者每 2-3 天更新一次进度。如 7 天内未审核，请随时提醒审核者或 vLLM 团队
- 审核后，如需修改，审核者会在 PR 上加 `action-required` 标签。请根据评论进行修改，并通知审核者重新审核
- 请及时回复所有评论。如有不明之处或不同意见，欢迎讨论
- 受限于算力资源，部分 CI 检查不会自动执行。当审核者认为 PR 可合并或需完整 CI 检查时，会加上 `ready` 标签

## 致谢

感谢你认真阅读本指南，并选择为 vLLM 贡献力量！你的每一份付出都让 vLLM 变得更好，助力整个社区共同成长！