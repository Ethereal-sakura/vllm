# 在 vLLM 开源 CI/CD 中更新 PyTorch 版本

vLLM 当前的政策是始终在 CI/CD 中使用最新的 PyTorch 正式版。每当有新的 [PyTorch 正式发布版本](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)上线时，尽早提交一个 PR 来升级 PyTorch，是标准操作流程。
由于 PyTorch 发布周期较长，这一升级过程并不简单。本文以 <https://github.com/vllm-project/vllm/pull/16859> 为例，梳理了常见的升级步骤，并列举了可能遇到的问题及解决方法。

## 测试 PyTorch 候选版本（RC）

在 PyTorch 官方正式发布后才在 vLLM 中升级，并不是理想选择。因为如果这时发现兼容性或其他问题，只能等下个 PyTorch 版本发布，或者在 vLLM 里临时用一些变通方法来规避。更好的做法，是在每次 PyTorch 发布之前，用候选版本（RC）提前测试 vLLM 的兼容性。

你可以从 [PyTorch 测试索引](https://download.pytorch.org/whl/test) 下载到 PyTorch 的候选版本。例如，想安装 `torch2.7.0+cu12.8` 的 RC，可以用如下命令：

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128
```

当最终的 RC 发布并可供测试时，社区会在 [PyTorch dev-discuss 论坛](https://dev-discuss.pytorch.org/c/release-announcements)进行公告。收到公告后，可以按照如下三步流程，提交一个 vLLM 的集成测试 PR：

1. 更新 [requirements 文件](https://github.com/vllm-project/vllm/tree/main/requirements)，将 `torch`、`torchvision` 和 `torchaudio` 指向新的版本。

2. 使用如下参数获取最终 RC 的安装包。常见平台包括 `cpu`、`cu128` 和 `rocm6.2.4`。

    ```bash
    --extra-index-url https://download.pytorch.org/whl/test/<PLATFORM>
    ```

3. 因为 vLLM 使用 `uv`，需要确保采用如下索引策略：

    - 通过环境变量设置：

    ```bash
    export UV_INDEX_STRATEGY=unsafe-best-match
    ```

    - 或直接用命令行参数：

    ```bash
    --index-strategy unsafe-best-match
    ```

如果在 PR 测试中遇到失败，可以在 vLLM 项目下提 issue，并 @ PyTorch 发布团队，以便讨论如何解决。

## 升级 CUDA 版本

PyTorch 的发布矩阵包含了多个 [CUDA 版本](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)，其中既有稳定版，也有实验版。由于限制，只有最新稳定的 CUDA 版本（比如 torch `2.7.1+cu126`）会上传到 PyPI。但 vLLM 有时需要其他 CUDA 版本，比如 12.8，以支持 Blackwell 芯片。
这就导致不能直接用 `pip install torch torchvision torchaudio` 一键安装。解决方案是在 vLLM 的 Dockerfile 里使用 `--extra-index-url`。

- 目前常用的索引地址如下：

| 平台      | `--extra-index-url` |
|-----------|---------------------|
| CUDA 12.8 | [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128) |
| CPU       | [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) |
| ROCm 6.2  | [https://download.pytorch.org/whl/rocm6.2.4](https://download.pytorch.org/whl/rocm6.2.4) |
| ROCm 6.3  | [https://download.pytorch.org/whl/rocm6.3](https://download.pytorch.org/whl/rocm6.3) |
| XPU       | [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu) |

- 需要将以下文件中的 CUDA 版本同步到你选择的版本，确保 vLLM 的发布包在 CI 中能够正确测试：
    - `.buildkite/release-pipeline.yaml`
    - `.buildkite/scripts/upload-wheels.sh`

## 解决 vLLM 构建时间过长的问题

每当用新的 PyTorch/CUDA 版本构建 vLLM 时，vLLM 的 sccache S3 bucket 里还没有对应的缓存，CI 上的构建任务可能会耗时超过 5 小时甚至超时。而且 vLLM 的 fastcheck 流程是只读的，无法自动填充缓存，所以多跑几次也没用。

目前官方正通过 <https://github.com/vllm-project/vllm/issues/17419> 这样的方案，从根本上解决构建时间过长的问题。但临时的 workaround 是，在 Buildkite 手动触发构建时，把 `VLLM_CI_BRANCH` 环境变量设置为 @khluu 提供的分支（比如 `VLLM_CI_BRANCH=khluu/long_build`）。这个分支有两个重要作用：

1. 把超时时间提升到 10 小时，避免构建过程被强制中断。
2. 允许构建好的文件写入 vLLM 的 sccache S3 bucket，提前热缓存，以加快后续构建速度。

<p align="center" width="100%">
    <img width="60%" src="https://github.com/user-attachments/assets/a8ff0fcd-76e0-4e91-b72f-014e3fdb6b94">
</p>

## 升级依赖项

vLLM 有一些依赖（比如 xFormers）也依赖 PyTorch，需要同步更新。与其等所有依赖都发布新版本，不如直接从源码编译，能更快打通升级流程。

### xFormers

```bash
export TORCH_CUDA_ARCH_LIST='7.5 8.0+PTX 9.0a'
MAX_JOBS=16 uv pip install --system \
    --no-build-isolation "git+https://github.com/facebookresearch/xformers@v0.0.32.post2"
```

## 分平台升级 vLLM

一次性在一个 PR 里升级所有 vLLM 平台并不现实，更推荐分平台逐步推进。由于 vLLM CI/CD 把不同平台的 requirements 和 Dockerfile 分开管理，可以灵活选择升级哪些平台。比如，升级 XPU 时需要用到 [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) 的专门版本。
像 <https://github.com/vllm-project/vllm/pull/16859> 就是优先完成了 CPU、CUDA、ROCm 平台的 PyTorch 2.7.0 升级，随后 <https://github.com/vllm-project/vllm/pull/17444> 又完成了 XPU 平台的升级。