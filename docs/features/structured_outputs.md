# 结构化输出

vLLM 支持生成结构化输出，可以选择 [xgrammar](https://github.com/mlc-ai/xgrammar) 或 [guidance](https://github.com/guidance-ai/llguidance) 作为后端。
本文将为你介绍几种生成结构化输出的常见方式，并提供相关示例。

!!! warning
    如果你还在使用以下已废弃的 API 字段，请按本文档的方式将代码更新为 `structured_outputs`：

    - `guided_json` -> `{"structured_outputs": {"json": ...}}` 或 `StructuredOutputsParams(json=...)`
    - `guided_regex` -> `{"structured_outputs": {"regex": ...}}` 或 `StructuredOutputsParams(regex=...)`
    - `guided_choice` -> `{"structured_outputs": {"choice": ...}}` 或 `StructuredOutputsParams(choice=...)`
    - `guided_grammar` -> `{"structured_outputs": {"grammar": ...}}` 或 `StructuredOutputsParams(grammar=...)`
    - `guided_whitespace_pattern` -> `{"structured_outputs": {"whitespace_pattern": ...}}` 或 `StructuredOutputsParams(whitespace_pattern=...)`
    - `structural_tag` -> `{"structured_outputs": {"structural_tag": ...}}` 或 `StructuredOutputsParams(structural_tag=...)`
    - `guided_decoding_backend` -> 请移除此字段

## 在线服务（OpenAI API）

你可以通过 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat](https://platform.openai.com/docs/api-reference/chat) API 生成结构化输出。

支持的参数如下，请作为额外参数添加：

- `choice`：输出会严格为指定选项之一。
- `regex`：输出会符合正则表达式规则。
- `json`：输出会符合指定的 JSON schema（模式）。
- `grammar`：输出会符合上下文无关语法（context free grammar）。
- `structural_tag`：在生成文本中，指定标签范围内遵循特定 JSON schema。

完整参数列表可参考 [OpenAI-Compatible Server](../serving/openai_compatible_server.md) 页面。

OpenAI-Compatible Server 默认支持结构化输出。你可通过设置 `--structured-outputs-config.backend` 参数指定后端，默认值为 `auto`，会根据请求自动选择合适的后端。你也可以指定特定后端以及相关选项，全部可选项可在 `vllm serve --help` 查看。

接下来我们以最简单的 `choice` 示例开始讲解：

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )
    model = client.models.list().data[0].id

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
    )
    print(completion.choices[0].message.content)
    ```

下一个例子展示了如何使用 `regex` 生成一个符合简单正则模板的邮箱地址：

??? code

    ```python
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
    )
    print(completion.choices[0].message.content)
    ```

结构化文本生成最实用的功能之一，就是可以生成包含预定义字段和格式的有效 JSON。
为此，可以通过两种方式使用 `json` 参数：

- 直接使用 [JSON Schema](https://json-schema.org/)
- 定义 [Pydantic 模型](https://docs.pydantic.dev/latest/)，并通过模型自动提取 JSON Schema（通常更易用）

下面这个例子展示了如何结合 Pydantic 模型和 `response_format` 参数生成结构化 JSON：

??? code

    ```python
    from pydantic import BaseModel
    from enum import Enum

    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    json_schema = CarDescription.model_json_schema()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "car-description",
                "schema": CarDescription.model_json_schema()
            },
        },
    )
    print(completion.choices[0].message.content)
    ```

!!! tip
    虽然不是强制要求，但通常建议在提示词中明确说明 JSON schema 及各字段的填写方式，这样能显著提升生成效果。

最后，我们来讲解 `grammar` 选项。它虽然较为复杂，但非常强大，可以让我们定义完整的语言格式（比如 SQL 查询语句）。其原理是利用上下文无关 EBNF 语法。

下面的例子定义了一个简化版 SQL 查询语法，并生成相应内容：

??? code

    ```python
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
    )
    print(completion.choices[0].message.content)
    ```

相关内容可参考：[完整示例](../examples/online_serving/structured_outputs.md)

## 推理输出

你还可以在 <project:#reasoning-outputs> 中结合结构化输出，用于推理类模型。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

注意，推理功能可与任何结构化输出参数共同使用。下面展示了结合 JSON schema 的用法：

??? code

    ```python
    from pydantic import BaseModel


    class People(BaseModel):
        name: str
        age: int


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the name and age of one random person.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people",
                "schema": People.model_json_schema()
            }
        },
    )
    print("reasoning_content: ", completion.choices[0].message.reasoning_content)
    print("content: ", completion.choices[0].message.content)
    ```

更多用法请参考：[完整示例](../examples/online_serving/structured_outputs.md)

## 实验性自动解析（OpenAI API）

本节介绍 OpenAI 针对 `client.chat.completions.create()` 方法的 beta 包装器，可自动和 Python 类型深度集成。

在撰写本文时（`openai==1.54.4`），这是 OpenAI 客户端库中的 "beta" 功能。代码参考见 [这里](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104)。

以下示例假设 vLLM 已使用 `vllm serve meta-llama/Llama-3.1-8B-Instruct` 启动。

下面是一个简单示例，展示如何使用 Pydantic 模型获取结构化输出：

??? code

    ```python
    from pydantic import BaseModel
    from openai import OpenAI

    class Info(BaseModel):
        name: str
        age: int

    client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
    model = client.models.list().data[0].id
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
        ],
        response_format=Info,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    print("Name:", message.parsed.name)
    print("Age:", message.parsed.age)
    ```

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

下面是一个更复杂的嵌套 Pydantic 模型示例，可用于分步骤展示数学解题过程：

??? code

    ```python
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful expert math tutor."},
            {"role": "user", "content": "Solve 8x + 31 = 2."},
        ],
        response_format=MathResponse,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    for i, step in enumerate(message.parsed.steps):
        print(f"Step #{i}:", step)
    print("Answer:", message.parsed.final_answer)
    ```

输出示例：

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
Step #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
Step #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
Step #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
Answer: x = -29/8
```

关于 `structural_tag` 的使用范例，请见：[examples/online_serving/structured_outputs](../../examples/online_serving/structured_outputs)

## 离线推理

离线推理同样支持结构化输出功能。
要使用此功能，可以在 `SamplingParams` 中配置 `StructuredOutputsParams` 类，例如：

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

这些参数用法与在线服务示例相同。下面是一个使用 `choice` 参数的离线推理示例：

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

更多内容请参考：[完整示例](../examples/online_serving/structured_outputs.md)
