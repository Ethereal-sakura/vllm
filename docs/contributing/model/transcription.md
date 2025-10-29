# 语音转文本（转录/翻译）功能支持

本文档将指导你如何通过实现 [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription]，为 vLLM 的转录（transcription）和翻译（translation）API 增加语音转文本（自动语音识别，ASR）模型的支持。
如需更多指导，请参考 [支持的模型](../../models/supported_models.md#transcription)。

## 更新 vLLM 基础模型

假定你已根据基础模型指南在 vLLM 中实现了自己的模型。你需要让模型继承 [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription] 接口，并实现如下类属性和方法。

### `supported_languages` 和 `supports_transcription_only`

声明模型支持的语言和功能：

- `supported_languages`（支持的语言）将在模型初始化时校验。
- 如果你的模型只支持基于音频的生成（如 Whisper，不支持文本生成），请将 `supports_transcription_only=True`。

??? code "supported_languages 和 supports_transcription_only"

    ```python
    from typing import ClassVar, Mapping, Literal
    import numpy as np
    import torch
    from torch import nn

    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.inputs.data import PromptType
    from vllm.model_executor.models.interfaces import SupportsTranscription
    
    class YourASRModel(nn.Module, SupportsTranscription):
        # ISO 639-1 语言代码到语言名称的映射
        supported_languages: ClassVar[Mapping[str, str]] = {
            "en": "English",
            "it": "Italian",
            # ... 可根据需要添加更多
        }
        
        # 如果你的模型仅支持基于音频的生成
        # （不支持纯文本生成），请启用此标志位。
        supports_transcription_only: ClassVar[bool] = True
    ```

通过实现 [get_speech_to_text_config][vllm.model_executor.models.interfaces.SupportsTranscription.get_speech_to_text_config] 提供 ASR 配置。

这个配置用于控制模型 API 的通用行为：

??? code "get_speech_to_text_config()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_speech_to_text_config(
            cls,
            model_config: ModelConfig,
            task_type: Literal["transcribe", "translate"],
        ) -> SpeechToTextConfig:
            return SpeechToTextConfig(
                sample_rate=16_000,
                max_audio_clip_s=30,
                # 如果模型或处理器已自行处理分片，则设为 None 禁用服务端分段
                min_energy_split_window_size=None,
            )
    ```

各字段的详细作用可见 [音频预处理与分段](#audio-preprocessing-and-chunking) 部分。

通过 [get_generation_prompt][vllm.model_executor.models.interfaces.SupportsTranscription.get_generation_prompt] 实现提示构建。服务端会将重采样后的音频波形和任务参数传递给你，你需要返回合法的 [PromptType][vllm.inputs.data.PromptType]。常见有两种模式：

#### 支持音频嵌入的多模态大语言模型（如 Voxtral, Gemma3n）

返回包含 `multi_modal_data`（音频数据）以及 `prompt` 字符串或 `prompt_token_ids` 的字典：

??? code "get_generation_prompt()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            audio: np.ndarray,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
            language: str | None,
            task_type: Literal["transcribe", "translate"],
            request_prompt: str,
            to_language: str | None,
        ) -> PromptType:
            # 示例：自定义指令式 prompt
            task_word = "Transcribe" if task_type == "transcribe" else "Translate"
            prompt = (
                "<start_of_turn>user\n"
                f"{task_word} this audio: <audio_soft_token>"
                "<end_of_turn>\n<start_of_turn>model\n"
            )

            return {
                "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                "prompt": prompt,
            }
    ```

    多模态输入的更多说明见 [Multi-Modal Inputs](../../features/multimodal_inputs.md)。

#### 编码器-解码器式音频模型（如 Whisper）

返回包含 `encoder_prompt` 和 `decoder_prompt` 的字典：

??? code "get_generation_prompt()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            audio: np.ndarray,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
            language: str | None,
            task_type: Literal["transcribe", "translate"],
            request_prompt: str,
            to_language: str | None,
        ) -> PromptType:
            if language is None:
                raise ValueError("Language must be specified")

            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (audio, stt_config.sample_rate),
                    },
                },
                "decoder_prompt": (
                    (f"<|prev|>{request_prompt}" if request_prompt else "")
                    + f"<|startoftranscript|><|{language}|>"
                    + f"<|{task_type}|><|notimestamps|>"
                ),
            }
            return cast(PromptType, prompt)
    ```

### `validate_language`（可选）

通过 [validate_language][vllm.model_executor.models.interfaces.SupportsTranscription.validate_language] 校验语言。

若你的模型需要指定语言，并希望有默认值，可重写此方法（如 Whisper 的做法）：

??? code "validate_language()"

    ```python
    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "未指定语言，默认使用 language='en'。如需转录其他语言音频，请在 TranscriptionRequest 中传递 `language` 字段。"
            )
            language = "en"
        return super().validate_language(language)
    ```

### `get_num_audio_tokens`（可选）

通过 [get_num_audio_tokens][vllm.model_executor.models.interfaces.SupportsTranscription.get_num_audio_tokens] 进行 token 数估算，用于流式统计。

快速返回音频时长与 token 数的估算，有助于提升流式使用统计的准确性：

??? code "get_num_audio_tokens()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_num_audio_tokens(
            cls,
            audio_duration_s: float,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
        ) -> int | None:
            # 若无法估算可返回 None，否则返回估算值
            return int(audio_duration_s * stt_config.sample_rate // 320)  # 示例
    ```

## 音频预处理与分段

API 服务端会负责基础的音频 I/O 及可选的分段操作，然后再构建 prompt：

- 重采样：输入音频会使用 `librosa` 重采样到 `SpeechToTextConfig.sample_rate` 指定的采样率。
- 分段：如果 `SpeechToTextConfig.allow_audio_chunking` 为 True，并且音频时长超过 `max_audio_clip_s`，服务端会将音频切分为重叠的分段，并为每段生成 prompt。重叠量由 `overlap_chunk_second` 控制。
- 能量感知分段：若设置了 `min_energy_split_window_size`，服务端会优先选择能量较低的区间分割，减少切分时对词语的干扰。

服务端相关逻辑如下：

??? code "_preprocess_speech_to_text()"

    ```python
    # vllm/entrypoints/openai/speech_to_text.py
    async def _preprocess_speech_to_text(...):
        language = self.model_cls.validate_language(request.language)
        ...
        y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (self.asr_config.allow_audio_chunking
                        and duration > self.asr_config.max_audio_clip_s)
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt,
                to_language=to_language,
            )
            prompts.append(prompt)
        return prompts, duration
    ```

## 任务自动注册与暴露

如果你的模型实现了接口，vLLM 会自动检测并注册转录相关支持：

```python
if supports_transcription(model):
    if model.supports_transcription_only:
        return ["transcription"]
    supported_tasks.append("transcription")
```

启用后，服务端会自动初始化转录和翻译的处理器：

```python
state.openai_serving_transcription = OpenAIServingTranscription(...) if "transcription" in supported_tasks else None
state.openai_serving_translation = OpenAIServingTranslation(...) if "transcription" in supported_tasks else None
```

只要你的模型类已在模型注册表中并实现了 `SupportsTranscription`，无需额外注册流程。

## 代码示例

- Whisper 编码器-解码器（仅音频）：[vllm/model_executor/models/whisper.py](../../../vllm/model_executor/models/whisper.py)
- Voxtral 解码器（音频嵌入+LLM）：[vllm/model_executor/models/voxtral.py](../../../vllm/model_executor/models/voxtral.py)
- Gemma3n 解码器+固定指令 prompt：[vllm/model_executor/models/gemma3n_mm.py](../../../vllm/model_executor/models/gemma3n_mm.py)

## API 调用测试

当你的模型实现了 `SupportsTranscription` 后，可以用如下方式测试接口（API 与 OpenAI 兼容）：

- 转录（ASR）：

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/transcriptions
    ```

- 翻译（源语言→英文，除非模型支持其它目标语言）：

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/translations
    ```

更多示例可参考 [examples/online_serving](../../../examples/online_serving)。

!!! note
    - 如果你的模型内部已自行处理音频分段（如处理器或编码器支持分片），请在返回的 `SpeechToTextConfig` 中设置 `min_energy_split_window_size=None`，禁用服务端分段。
    - 实现 `get_num_audio_tokens` 可提升流式统计（`prompt_tokens`）的准确性，无需额外前向推理。
    - 若支持多语言，请确保 `supported_languages` 与模型实际能力保持一致。
