加载模型权重（Model weights）到 GPU —— 使用 fastsafetensors
===================================================================

通过 fastsafetensors 库，可以利用 GPU 直连存储（GPU direct storage）将模型权重直接加载到 GPU 内存。更多细节请参阅 [他们的 GitHub 仓库](https://github.com/foundation-model-stack/fastsafetensors) 

要开启此功能，请在命令行中添加参数 `--load-format fastsafetensors`
