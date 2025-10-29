# --8<-- [start:installation]

vLLM 目前在搭载 Apple Silicon 的 macOS 上提供了实验性支持。现在，用户需要从源码自行编译，才能在 macOS 上原生运行。

目前 macOS 平台上的 CPU 实现支持 FP32 和 FP16 两种数据类型。

!!! warning
    目前没有适用于该设备的预编译轮子（wheels）或镜像（images），你必须从源码编译 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：`macOS Sonoma` 或更高版本
- SDK：需安装带命令行工具的 `XCode 15.4` 或更高版本
- 编译器：`Apple Clang >= 15.0.0`

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

在安装好 XCode 及其命令行工具（包含 Apple Clang）后，请依次执行以下命令，从源码编译并安装 vLLM。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

!!! note
    在 macOS 上，`VLLM_TARGET_DEVICE` 会自动设置为 `cpu`，目前仅支持该设备。

!!! example "故障排查"
    如果编译时遇到如下错误，提示无法找到标准 C++ 头文件，可以尝试卸载并重新安装
    [Xcode 命令行工具](https://developer.apple.com/download/all/)。

    ```text
    [...] fatal error: 'map' file not found
            1 | #include <map>
                |          ^~~~~
        1 error generated.
        [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o

    [...] fatal error: 'cstddef' file not found
            10 | #include <cstddef>
                |          ^~~~~~~~~
        1 error generated.
    ```

    ---

    如果编译失败，出现类似 C++11/C++17 兼容性错误，如下所示，说明构建系统默认使用了较老的 C++ 标准：

    ```text
    [...] error: 'constexpr' is not a type
    [...] error: expected ';' before 'constexpr'
    [...] error: 'constexpr' does not name a type
    ```

    **解决方法**：你的编译器可能正在使用较老的 C++ 标准。请编辑 `cmake/cpu_extension.cmake`，在 `set(CMAKE_CXX_STANDARD_REQUIRED ON)` 之前添加一行 `set(CMAKE_CXX_STANDARD 17)`。

    检查编译器 C++ 标准支持的方法如下：
    ```bash
    clang++ -std=c++17 -pedantic -dM -E -x c++ /dev/null | grep __cplusplus
    ```
    在 Apple Clang 16 上应该看到：`#define __cplusplus 201703L`

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]