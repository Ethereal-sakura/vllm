当你通过命令行传递 JSON 参数时，以下两种方式是等效的：

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

此外，列表（List）元素也可以通过单独的参数传递，使用 `+` 符号：

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`