# CI 失败处理

如果我的 PR（Pull Request）触发了 CI（持续集成 Continuous Integration）任务失败，但我认为不是我的原因导致的，该怎么办？

- 请先查看当前 CI 测试失败的看板：  
  👉 [CI 失败看板](https://github.com/orgs/vllm-project/projects/20)

- 如果你的失败**已经在列表中**，那很可能与你的 PR 无关。当然，非常欢迎你参与修复！
    - 可以在相关问题下留言，提供更多失败实例的链接。
    - 给已有的问题点个 👍，表示有更多人受到影响。

- 如果你的失败**没有在列表中**，请**新建一个 issue** 进行反馈。

## 如何新建 CI 测试失败的 Issue

- **提交 Bug 报告：**  
    👉 [新建 CI 失败报告](https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml)

- **标题请使用以下格式：**

    ```text
    [CI Failure]: failing-test-job - regex/matching/failing:test
    ```

- **环境字段建议填写：**

    ```text
    Still failing on main as of commit abcdef123
    ```

- **描述中请包含出错的测试项：**

    ```text
    FAILED failing/test.py:failing_test1 - 失败描述
    FAILED failing/test.py:failing_test2 - 失败描述
    https://github.com/orgs/vllm-project/projects/20
    https://github.com/vllm-project/vllm/issues/new?template=400-bug-report.yml
    FAILED failing/test.py:failing_test3 - 失败描述
    ```

- **请附上日志**（可折叠的日志示例）：
    <details>
    <summary>日志：</summary>

    ```text
    ERROR 05-20 03:26:38 [dump_input.py:68] Dumping input data
    --- Logging error ---  
    Traceback (most recent call last):  
      File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 203, in execute_model  
        return self.model_executor.execute_model(scheduler_output)
    ...
    FAILED failing/test.py:failing_test1 - 失败描述
    FAILED failing/test.py:failing_test2 - 失败描述
    FAILED failing/test.py:failing_test3 - 失败描述
    ```

    </details>

## 日志处理

先从 Buildkite 下载完整日志文件到本地。

去除时间戳和颜色高亮：

[.buildkite/scripts/ci-clean-log.sh](../../../.buildkite/scripts/ci-clean-log.sh)

```bash
./ci-clean-log.sh ci.log
```

可以使用 [wl-clipboard](https://github.com/bugaevc/wl-clipboard) 工具快速复制粘贴：

```bash
tail -525 ci_build.log | wl-copy
```

## 如何排查 CI 测试失败

1. 打开 👉 [Buildkite 主分支页面](https://buildkite.com/vllm/ci/builds?branch=main)
2. 通过二分法查找首次出现该问题的构建。  
3. 把你的排查结果补充到 GitHub issue 里。  
4. 如果定位到可能相关的 PR，请在 issue 里提及并 @ 相关贡献者。

## 如何复现失败

CI 测试失败有时可能是偶发问题（flaky）。可以用 bash 循环多次运行：

[.buildkite/scripts/rerun-test.sh](../../../.buildkite/scripts/rerun-test.sh)

```bash
./rerun-test.sh tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]
```

## 提交修复 PR

如果你要提交 PR 修复 CI 问题：

- 请在 PR 描述中关联 issue：
  添加 `Closes #12345` 到 PR 描述里。
- 添加 `ci-failure` 标签：
  这样可以便于在 [CI 失败看板](https://github.com/orgs/vllm-project/projects/20) 跟踪。

## 其他资源

- 🔍 [主分支测试稳定性](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&order=ASC&sort_by=reliability)
- 🧪 [最新 Buildkite CI 运行](https://buildkite.com/vllm/ci/builds?branch=main)

## 日常分诊

使用 [Buildkite 分析（近 2 天）](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&period=2days) 可以：

- 识别主分支（main）上最近的测试失败。
- 排除 PR 上的有效测试失败。
- （可选）忽略稳定性为 0% 的测试。

建议与 [CI 失败看板](https://github.com/orgs/vllm-project/projects/20) 对比查看。