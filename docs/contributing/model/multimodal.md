# 多模态支持

本文将带你逐步实现如何扩展基础模型，以便其支持[多模态输入](../../features/multimodal_inputs.md)。

## 1. 更新基础 vLLM 模型

这里假设你已经按照[这篇文档](basic.md)的指引，在 vLLM 中实现了模型。
接下来按照如下方式进一步完善模型：

- 实现[get_placeholder_str][vllm.model_executor.models.interfaces.SupportsMultiModal.get_placeholder_str]，用于定义在文本 prompt 中用来表示多模态内容的占位符字符串。这个占位符要和模型的聊天模板保持一致。

    ??? code

        ```python
        class YourModelForImage2Seq(nn.Module):
            ...

            @classmethod
            def get_placeholder_str(cls, modality: str, i: int) -> str | None:
                if modality.startswith("image"):
                    return "<image>"

                raise ValueError("Only image modality is supported")
        ```

- 在[forward][torch.nn.Module.forward]函数中，为每个与多模态输入对应的张量预留一个关键字参数，例如：

  ```diff
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
  +     pixel_values: torch.Tensor,
    ) -> SamplerOutput:
  ```
  
  更方便的做法是直接在[forward][torch.nn.Module.forward]方法中使用 `**kwargs`，然后从中获取多模态输入的关键字参数。

- 实现[get_multimodal_embeddings][vllm.model_executor.models.interfaces.SupportsMultiModal.get_multimodal_embeddings]，该方法需通过多模态 tokenizer，将多模态输入转化为 embedding。下面给出典型实现模板，具体可结合实际需要调整。

    ??? code

        ```python
        class YourModelForImage2Seq(nn.Module):
            ...

            def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:
                assert self.vision_encoder is not None
                image_features = self.vision_encoder(image_input)
                return self.multi_modal_projector(image_features)

            def get_multimodal_embeddings(
                self,
                **kwargs: object,
            ) -> MultiModalEmbeddings | None:
                # 校验多模态输入参数
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is None:
                    return None

                # 编码器和 projector 处理多模态输入
                vision_embeddings = self._process_image_input(image_input)
                return vision_embeddings
        ```

!!! important
    返回的 `multimodal_embeddings` 必须是如下两种格式之一：  
    - 形状为 `(num_items, feature_size, hidden_size)` 的三维 **[torch.Tensor][]**  
    - 或是一个包含若干形状为 `(feature_size, hidden_size)` 的二维 **[torch.Tensor][]** 的列表/元组  
    这样才能通过 `multimodal_embeddings[i]` 正确获取第 `i` 个多模态数据（例如图片）的 embedding。

!!! note
    vLLM 默认会根据输入处理中[PlaceholderRange][vllm.multimodal.inputs.PlaceholderRange]的信息，将多模态 embedding 合并到文本 embedding 中。具体合并逻辑见[get_input_embeddings][vllm.model_executor.models.interfaces.SupportsMultiModal.get_input_embeddings]。

    如需更复杂的 embedding 合并逻辑，可重写此方法。

- 实现[get_language_model][vllm.model_executor.models.interfaces.SupportsMultiModal.get_language_model]方法，为底层语言模型提供稳定的访问接口。

    ```python
    class YourModelForImage2Seq(nn.Module):
        ...

        def get_language_model(self) -> torch.nn.Module:
            # 根据你的实现修改 language_model
            return self.language_model
    ```

- 完成上述工作后，确保模型类继承了[SupportsMultiModal][vllm.model_executor.models.interfaces.SupportsMultiModal]接口。

  ```diff
  + from vllm.model_executor.models.interfaces import SupportsMultiModal

  - class YourModelForImage2Seq(nn.Module):
  + class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
  ```

!!! note
    模型类不一定非得叫 `*ForCausalLM`。你可以参考[HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/model_doc/auto#multimodal)中的一些多模态模型示例。

## 2. 指定处理信息

接下来，创建 [BaseProcessingInfo][vllm.multimodal.processing.BaseProcessingInfo] 的子类，用于提供与 HF 处理相关的基本信息。

### 支持的输入数量限制

需要重写 [get_supported_mm_limits][vllm.multimodal.processing.BaseProcessingInfo.get_supported_mm_limits] 抽象方法，返回模型支持的每种模态的最大输入数量。

比如，模型支持任意数量图片，但每次只支持一个视频：

```python
def get_supported_mm_limits(self) -> Mapping[str, int | None]:
    return {"image": None, "video": 1}
```

## 3. 指定虚拟输入（dummy inputs）

然后，继承 [BaseDummyInputsBuilder][vllm.multimodal.profiling.BaseDummyInputsBuilder]，用于构造 HF 处理与内存分析所需的虚拟输入。

### 用于内存分析

重写 [get_dummy_text][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text] 和 [get_dummy_mm_data][vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_mm_data] 抽象方法，生成用于内存分析的虚拟输入。这样可以确保 vLLM 为模型预留足够的内存。

通常假设内存占用与 token 数量有关，所以虚拟输入应尽量产生最多的输出 embedding，即占位符特征 token 的数量。

=== "基础示例：LLaVA"

    查看 HF 的 `LlavaForConditionalGeneration` 代码：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L530-L544
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        ```

    每幅图像的占位符特征 token 数量为 `image_features.shape[1]`。
    `image_features` 在 `get_image_features` 方法中计算：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L290-L300
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
        ```

    可以看出，`image_features.shape[1]` 取决于 vision tower（比如[`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)模型中的`CLIPVisionModel`）输出的 hidden states 的第二维。
    由于 attention 不会改变序列长度，所以序列长度由 `CLIPVisionTransformer` 的初始 hidden states 决定。

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L1094-L1102
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    ```

    序列长度可在 `CLIPVisionEmbeddings` 代码中找到：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L247-L257
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        ```

    可以推出 `embeddings.shape[1] == self.num_positions`，其中

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L195-L196
    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.num_positions = self.num_patches + 1
    ```

    总结下来，单张图片的特征 token 数量可这样计算：

    ??? code

        ```python
        def get_num_image_tokens(
            self,
            *,
            image_width: int,
            image_height: int,
        ) -> int:
            hf_config = self.get_hf_config()
            hf_processor = self.get_hf_processor()

            image_size = hf_config.vision_config.image_size
            patch_size = hf_config.vision_config.patch_size

            num_image_tokens = (image_size // patch_size) ** 2 + 1
            if hf_processor.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            return num_image_tokens
        ```

    注意，图片 token 数量与实际图片宽高无关，只需用 dummy 的 `image_size` 计算多模态 profiling 数据即可：

    ??? code

        ```python
        # 通常实现为模型的 `BaseProcessingInfo` 子类方法，这里为了简化直接写出
        def get_image_size_with_most_features(self) -> ImageSize:
            hf_config = self.get_hf_config()
            width = height = hf_config.image_size
            return ImageSize(width=width, height=height)

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions] | None = None,
        ) -> MultiModalDataDict:
            num_images = mm_counts.get("image", 0)

            target_width, target_height = \
                self.info.get_image_size_with_most_features()

            image_overrides = mm_options.get("image") if mm_options else None

            return {
                "image":
                self._get_dummy_images(width=target_width,
                                    height=target_height,
                                    num_images=num_images,
                                    overrides=image_overrides)
            }
        ```

    对于文本，只需根据模型配置，将多模态图片 token 扩展到所需图片数量：

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images
    ```

=== "不含占位符的模型：Fuyu"

    查看 HF 的 `FuyuForCausalLM` 代码：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/modeling_fuyu.py#L311-L322
        if image_patches is not None and past_key_values is None:
            patch_embeddings = [
                self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype))
                .squeeze(0)
                .to(inputs_embeds.device)
                for patch in image_patches
            ]
            inputs_embeds = self.gather_continuous_embeddings(
                word_embeddings=inputs_embeds,
                continuous_embeddings=patch_embeddings,
                image_patch_input_indices=image_patches_indices,
            )
        ```

    批次中第 `i` 个图片的特征 token 数量为 `patch_embeddings[i].shape[0]`，等价于 `image_patches[i].shape[0]`，即 `num_total_patches`。

    与 LLaVA 不同，Fuyu 并未在建模文件中定义 patch 数量。我们可以在预处理文件里查找更多信息。

    图片输出由 `FuyuImageProcessor.preprocess` 和 `FuyuImageProcessor.preprocess_with_tokenizer_info` 生成。

    在 `FuyuImageProcessor.preprocess` 中，图片首先 resize、pad 到目标大小，返回 resize（pad 之前）的尺寸作为元数据。

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L541-L544
        image_encoding = self.image_processor.preprocess(images, **output_kwargs["images_kwargs"])
        batch_images = image_encoding["images"]
        image_unpadded_heights = image_encoding["image_unpadded_heights"]
        image_unpadded_widths = image_encoding["image_unpadded_widths"]

        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L480-L
        if do_resize:
            batch_images = [
                [self.resize(image, size=size, input_data_format=input_data_format) for image in images]
                for images in batch_images
            ]

        image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]

        if do_pad:
            batch_images = [
                [
                    self.pad_image(
                        image,
                        size=size,
                        mode=padding_mode,
                        constant_values=padding_value,
                        input_data_format=input_data_format,
                    )
                    for image in images
                ]
                for images in batch_images
            ]
        ```

    在 `FuyuImageProcessor.preprocess_with_tokenizer_info` 中，图片会依据元数据被切成 patch：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L417-L425
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )

        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L638-L658
        image_height, image_width = image.shape[1], image.shape[2]
        if variable_sized:  # variable_sized=True
            new_h = min(
                image_height,
                math.ceil(image_unpadded_h[batch_index, subseq_index] / patch_height) * patch_height,
            )
            new_w = min(
                image_width,
                math.ceil(image_unpadded_w[batch_index, subseq_index] / patch_width) * patch_width,
            )
            image = image[:, :new_h, :new_w]
            image_height, image_width = new_h, new_w

        num_patches = self.get_num_patches(image_height=image_height, image_width=image_width)
        tensor_of_image_ids = torch.full(
            [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
        )
        patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
        assert num_patches == patches.shape[0]
        ```

    patch 数量由 `FuyuImageProcessor.get_num_patches` 定义：

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L552-L562
        patch_size = patch_size if patch_size is not None else self.patch_size
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")

        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        ```

    这些 patch 就是占位 token（