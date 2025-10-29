# 多模态输入

本页将指导你如何在 vLLM 中为[多模态模型](../models/supported_models.md#list-of-multimodal-language-models)传递多模态输入。

!!! note
    我们正在不断完善多模态支持。你可以通过[这个 RFC](https://github.com/vllm-project/vllm/issues/4194)了解即将发布的功能，
    如果有任何反馈或功能需求，欢迎[在 GitHub 提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)。

!!! tip
    在部署多模态模型时，建议通过设置 `--allowed-media-domains` 参数限制 vLLM 可以访问的域名，防止访问任意端点，规避可能的服务端请求伪造（SSRF）攻击风险。你可以为此参数提供一个域名列表，例如：`--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`

    此外，可以设置 `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`，防止通过 HTTP 重定向绕过域名限制。

    当你在容器环境下运行 vLLM，且 vLLM pod 可能拥有内网访问权限时，这一限制尤为重要。

## 离线推理

传递多模态数据时，请按照 [vllm.inputs.PromptType][] 的以下结构：

- `prompt`：提示词需遵循 HuggingFace 官方文档格式。
- `multi_modal_data`：这是一个字典，格式参考 [vllm.multimodal.inputs.MultiModalDataDict][]。

### 缓存用稳定 UUID（multi_modal_uuids）

使用多模态输入时，vLLM 默认会对每个媒体内容进行哈希，实现跨请求的缓存。你也可以传递 `multi_modal_uuids`，为每个媒体项指定稳定的 ID，让缓存可复用，无需重新哈希原始内容。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL 示例，包含两张图片
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_a = Image.open("/path/to/a.jpg")
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [img_a, img_b]},
        # 提供稳定的缓存 ID
        # 要求（示例已满足）：
        #  - 每种模态都需包含在 multi_modal_data 中
        #  - 列表需与数据项数量一致
        #  - 用 None 表示该项仍采用内容哈希
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

如果你预期某些媒体项会命中缓存，也可以只发送 UUID，而跳过媒体数据。注意：如果跳过的媒体项没有对应 UUID，或 UUID 未命中缓存，请求会失败。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL 示例，包含两张图片
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [None, img_b]},
        # 由于 img_a 预计已被缓存，可不传实际图片
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

!!! warning
    如果多模态处理器缓存和前缀缓存都被关闭，用户自定义的 `multi_modal_uuids` 会被忽略。

### 图片输入

你可以像下面这样，将单张图片传递到多模态字典的 `'image'` 字段：

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # 具体 prompt 格式请参考 HuggingFace 仓库
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # 使用 PIL.Image 加载图片
    image = PIL.Image.open(...)

    # 单条推理
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # 批量推理
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完整示例见：[examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)

如果要在同一个文本 prompt 中插入多张图片，可以直接传递图片列表：

??? code

    ```python
    from vllm import LLM

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # 加载 Phi-3.5-vision 必须开启
        max_model_len=4096,      # 否则可能无法在小显卡上运行
        limit_mm_per_prompt={"image": 2},  # 每次最多支持 2 张图片
    )

    # 具体 prompt 格式请参考 HuggingFace 仓库
    prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # 用 PIL.Image 加载图片
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [image1, image2]},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完整示例见：[examples/offline_inference/vision_language_multi_image.py](../../examples/offline_inference/vision_language_multi_image.py)

如果你使用 [LLM.chat](../models/generative_models.md#llmchat) 方法，可以在消息内容中直接传递图片，支持多种格式：图片 URL、PIL Image 对象或预计算的 embedding：

```python
from vllm import LLM
from vllm.assets.image import ImageAsset

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image_url = "https://picsum.photos/id/32/512/512"
image_pil = ImageAsset('cherry_blossom').pil_image
image_embeds = torch.load(...)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
            {
                "type": "image_pil",
                "image_pil": image_pil,
            },
            {
                "type": "image_embeds",
                "image_embeds": image_embeds,
            },
            {
                "type": "text",
                "text": "What's in these images?",
            },
        ],
    },
]

# 推理并输出结果
outputs = llm.chat(conversation)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

多图片输入还可以扩展为视频字幕生成。比如 [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 支持视频：

??? code

    ```python
    from vllm import LLM

    # 指定每个视频最多 4 帧，可根据实际需求调整
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

    # 构造请求体
    video_frames = ... # 加载你的视频数据，确保帧数与前面设置一致
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this set of frames. Consider the frames to be a part of the same video.",
            },
        ],
    }
    for i in range(len(video_frames)):
        base64_image = encode_image(video_frames[i]) # base64 编码
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)

    # 推理并输出结果
    outputs = llm.chat([message])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

#### 自定义 RGBA 背景色

加载带透明通道的 RGBA 图片时，vLLM 会自动转换为 RGB 格式。默认情况下，透明像素会填充为白色背景。你可以通过 `media_io_kwargs` 的 `rgba_background_color` 参数自定义背景色。

??? code

    ```python
    from vllm import LLM

    # 默认白色背景（无需配置）
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # 自定义黑色背景，适合深色主题
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 0]}},
    )

    # 自定义品牌色（如蓝色背景）
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 255]}},
    )
    ```

!!! note
    - `rgba_background_color` 接受 RGB 值，格式为列表 `[R, G, B]` 或元组 `(R, G, B)`，每个值范围是 0-255
    - 此设置只对带透明通道的 RGBA 图片有效，RGB 图片不受影响
    - 未设置时，默认使用白色背景 `(255, 255, 255)`，以兼容旧版本

### 视频输入

你可以直接将 NumPy 数组列表传递到多模态字典的 `'video'` 字段，
也可以传递 `'torch.Tensor'` 对象，例如 Qwen2.5-VL 的用法如下：

??? code

    ```python
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    video_path = "https://content.pexels.com/videos/free-videos.mp4"

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(max_tokens=1024)

    video_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this video."},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]

    messages = video_messages
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

    !!! note
        'process_vision_info' 仅适用于 Qwen2.5-VL 及其同类模型。

完整示例见：[examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)

### 音频输入

你可以将元组 `(array, sampling_rate)` 传递到多模态字典的 `'audio'` 字段。

完整示例见：[examples/offline_inference/audio_language.py](../../examples/offline_inference/audio_language.py)

### 嵌入输入

要直接将已预计算的 embedding（如图片、视频或音频的特征）输入到语言模型，
只需将形状为 `(num_items, feature_size, hidden_size of LM)` 的张量传递到对应的多模态字段。

你需要通过 `enable_mm_embeds=True` 开启此功能。

!!! warning
    如果传递的 embedding 形状不正确，vLLM 引擎可能会崩溃。
    仅对可信用户启用此选项！

??? code

    ```python
    from vllm import LLM

    # 输入图片 embedding 进行推理
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", enable_mm_embeds=True)

    # 具体 prompt 格式请参考 HuggingFace 仓库
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # 单图片 embedding
    # 形状为 (1, image_feature_size, hidden_size of LM) 的 torch.Tensor
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

对于 Qwen2-VL 和 MiniCPM-V，还可以在 embedding 字段旁边传递附加参数：

??? code

    ```python
    # 根据模型构造 prompt
    prompt = ...

    # 多图片 embedding
    # 形状为 (num_images, image_feature_size, hidden_size of LM) 的 torch.Tensor
    image_embeds = torch.load(...)

    # Qwen2-VL
    llm = LLM(
        "Qwen/Qwen2-VL-2B-Instruct",
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_grid_thw 用于计算位置编码
            "image_grid_thw": torch.load(...),  # 形状为 (1, 3) 的 torch.Tensor
        }
    }

    # MiniCPM-V
    llm = LLM(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_sizes 用于计算分块图片细节
            "image_sizes": [image.size for image in images],  # 图片尺寸列表
        }
    }

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

## 在线服务

我们的 OpenAI 兼容服务端支持多模态数据，通过 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 接收。
你也可以为媒体输入提供 UUID，以便缓存跨请求复用。

!!! important
    使用 Chat Completions API 时**必须**配置对话模板。
    对于 HF 格式模型，默认模板定义在 `chat_template.json` 或 `tokenizer_config.json` 中。

    若没有默认模板，将优先查找 [vllm/transformers_utils/chat_templates/registry.py](../../vllm/transformers_utils/chat_templates/registry.py) 内置备选模板。
    仍未找到则会报错，此时需用 `--chat-template` 参数手动指定模板。

    某些模型可在 [examples](../../examples) 下找到备用模板。
    比如 VLM2Vec 使用的是 [examples/template_vlm2vec_phi3v.jinja](../../examples/template_vlm2vec_phi