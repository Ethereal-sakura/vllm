# Fused MoE Kernel 特性

本文档旨在为不同类型的 MoE（专家混合，Mixture of Experts）核（包括模块化和非模块化）提供一个全面的概览，帮助你根据实际需求更容易地选择合适的内核。同时，也介绍了模块化核所用到的 all2all 通信后端。

## Fused MoE 模块化 All2All 通信后端

在 `FusedMoE` 层中，实现专家并行（Expert Parallelism, EP）时可选用多种 all2all 通信后端。每种 all2all 后端都通过不同的 `FusedMoEPrepareAndFinalize` 子类进行对接。

下表对每种后端的关键特性进行了说明，包括激活格式、支持的量化方案以及是否支持异步。

输出激活格式（standard 或 batched）对应 `FusedMoEPrepareAndFinalize` 子类的 prepare 步骤输出，finalize 步骤则要求相同格式。所有后端的 `prepare` 方法都要求 standard 格式输入，所有 `finalize` 方法会输出 standard 格式。关于格式的更多细节，可以参考 [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) 文档。

量化类型与格式列出每个 `FusedMoEPrepareAndFinalize` 类支持的量化方案。量化步骤可以发生在分派前或分派后，具体取决于后端支持的格式。例如，deepep_high_throughput 只支持块量化的 fp8 格式，其它格式会在分派后以更高精度进行量化。每个后端的 prepare 步骤输出都是量化后的类型，finalize 步骤通常要求输入与原始激活类型相同，例如原始输入为 bfloat16 且量化方案为 fp8 配合 per-tensor scales，那么 prepare 返回 fp8/per-tensor scale 激活，finalize 则接收 bfloat16 激活。更多不同步骤激活类型与格式的示意，请参考 [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) 文档。如果未指定量化类型，内核默认使用 float16 或 bfloat16。

异步后端支持 DBO（Dual Batch Overlap，双批次重叠）和专家共享重叠（即在 combine 阶段进行 shared experts 计算）。

部分模型在 topk==1 时（如 llama），需要在输入激活上应用 topk 权重，而不是输出激活。对于模块化核，由 `FusedMoEPrepareAndFinalize` 子类支持此功能；对于非模块化核，则需要专家函数自行处理此标志。

除非特别说明，后端通过 `VLLM_ALL2ALL_BACKEND` 控制。除 `flashinfer` 外，所有后端仅支持 EP+DP 或 EP+TP 组合。`Flashinfer` 则可用于 EP 或纯 DP。

<style>
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}
</style>

| Backend                               | Output act. format | Quant. types    | Quant. format          | Async | Apply Weight On Input | Sub-class                                                                                                                                                     |
|---------------------------------------|--------------------|-----------------|------------------------|-------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| naive                                 | standard           | all<sup>1</sup> | G,A,T                  | N     | <sup>6</sup>          | [layer.py][vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_impl]                                                                                  |
| pplx                                  | batched            | fp8,int8        | G,A,T                  | Y     | Y                     | [`PplxPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.pplx_prepare_finalize.PplxPrepareAndFinalize]                                                 |
| deepep_high_throughput                | standard           | fp8             | G(128),A,T<sup>2</sup> | Y     | Y                     | [`DeepEPLLPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize.DeepEPLLPrepareAndFinalize]                                    |
| deepep_low_latency                    | batched            | fp8             | G(128),A,T<sup>3</sup> | Y     | Y                     | [`DeepEPHTPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize.DeepEPHTPrepareAndFinalize]                                    |
| flashinfer_all2allv                   | standard           | nvfp4,fp8       | G,A,T                  | N     | N                     | [`FlashInferAllToAllMoEPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize.FlashInferAllToAllMoEPrepareAndFinalize] |
| flashinfer<sup>4</sup>                | standard           | nvfp4,fp8       | G,A,T                  | N     | N                     | [`FlashInferCutlassMoEPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize.FlashInferCutlassMoEPrepareAndFinalize]   |
| flashinfer<sup>4</sup>                | standard           | nvfp4,fp8       | G,A,T                  | N     | N                     | [`FlashInferCutlassMoEPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize.FlashInferCutlassMoEPrepareAndFinalize]   |
| MoEPrepareAndFinalizeNoEP<sup>5</sup> | standard           | fp8,int8        | G,A,T                  | N     | Y                     | [`MoEPrepareAndFinalizeNoEP`][vllm.model_executor.layers.fused_moe.prepare_finalize.MoEPrepareAndFinalizeNoEP]                                                |
| BatchedPrepareAndFinalize<sup>5</sup> | batched            | fp8,int8        | G,A,T                  | N     | Y                     | [`BatchedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.fused_batched_moe.BatchedPrepareAndFinalize]                                               |

!!! info "表格注释"
    1. 支持所有类型：mxfp4、nvfp4、int4、int8、fp8
    2. A,T 量化发生在分派之后
    3. 所有量化都在分派后进行
    4. 通过不同的环境变量控制（`VLLM_FLASHINFER_MOE_BACKEND`，可选 "throughput" 或 "latency"）
    5. 这是一个无实际分派的调度器，可与任意模块化 experts 搭配，生成无需分派或 combine 的模块化内核。不能通过环境变量选择，通常用于测试或适配 expert 子类到 `fused_experts` API。
    6. 依赖于 experts 的具体实现。

    ---

    - G - 按组分组
    - G(N) - 按块大小 N 分组
    - A - 针对每个激活 token
    - T - 针对每个张量

模块化内核由以下 `FusedMoEMethodBase` 类支持：

- [`ModelOptFp8MoEMethod`][vllm.model_executor.layers.quantization.modelopt.ModelOptFp8MoEMethod]
- [`Fp8MoEMethod`][vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod]
- [`CompressedTensorsW4A4MoeMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsW4A4MoeMethod]
- [`CompressedTensorsW8A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsW8A8Fp8MoEMethod]
- [`Mxfp4MoEMethod`][vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod]
- [`UnquantizedFusedMoEMethod`][vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod]

## Fused MoE Experts 内核

针对不同的量化类型和硬件架构，MoE experts 实现了多种内核。大多数都遵循 Triton 基础 [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts] 函数的通用 API。许多实现带有模块化内核适配器，可与兼容的 all2all 后端配合使用。下表列出了每种 experts 内核及其具体特性。

每个内核要求输入为支持的激活格式。有些内核支持 standard 和 batched 格式，分别有不同的入口，比如 `TritonExperts` 和 `BatchedTritonExperts`。目前 batched 格式主要用于与特定 all2all 后端（如 `pplx`、`DeepEPLLPrepareAndFinalize`）配合。

和后端类似，每个 experts 内核也只支持特定的量化格式。非模块化 experts 会在内部进行量化，而模块化 experts 要求输入激活已是量化格式。两者输出均为原始激活类型。

每个 experts 内核支持一种或多种激活函数（如 silu、gelu），这些函数会应用在中间结果上。

同样，有些 experts 支持在输入激活上应用 topk 权重。表中相关列只适用于非模块化 experts。

大多数 experts 变体都提供了等价的模块化接口，通常是 `FusedMoEPermuteExpertsUnpermute` 的子类。

要与某个 `FusedMoEPrepareAndFinalize` 子类配合使用，MoE 内核需满足激活格式、量化类型和量化格式的兼容性。

| Kernel                       | Input act. format     | Quant. types     | Quant. format | Activation function                                         | Apply Weight On Input | Modular | Source                                                                                                                                                                                                                                                                                                      |
|------------------------------|-----------------------|------------------|---------------|-------------------------------------------------------------|-----------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| triton                       | standard              | all<sup>1</sup>  | G,A,T         | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y                     | Y       | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts]                                                                                                                                        |
| triton (batched)             | batched               | all<sup>1</sup>  | G,A,T         | silu, gelu                                                  | <sup>6</sup>          | Y       | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.fused_batched_moe.BatchedTritonExperts]                                                                                                                                                                                                       |
| deep gemm                    | standard,</br>batched | fp8              | G(128),A,T    | silu, gelu                                                  | <sup>6</sup>          | Y       | [`deep_gemm_moe_fp8`][vllm.model_executor.layers.fused_moe.deep_gemm_moe.deep_gemm_moe_fp8],</br>[`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.deep_gemm_moe.DeepGemmExperts],</br>[`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4                  | standard,</br>batched | nvfp4            | A,T           | silu                                                        | Y                     | Y       | [`cutlass_moe_fp4`][vllm.model_executor.layers.fused_moe.cutlass_moe.cutlass_moe_fp4],</br>[`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassExpertsFp4]                                                                                                                        |
| cutlass_fp8                  | standard,</br>batched | fp8              | A,T           | silu, gelu                                                  | Y                     | Y       | [`cutlass_moe_fp8`][vllm.model_executor.layers.fused_moe.cutlass_moe.cutlass_moe_fp8],</br>[`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassExpertsFp8],</br>[`CutlasBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassBatchedExpertsFp8]            |
| flashinfer                   | standard              | nvfp4,</br>fp8   | T             | <sup>5</sup>                                                | N                     | Y       | [`flashinfer_cutlass_moe_fp4`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe.flashinfer_cutlass_moe_fp4],</br>[`FlashInferExperts`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe.FlashInferExperts]                                                                            |
| gpt oss triton               | standard              | N/A              | N/A           | <sup>5</sup>                                                | Y                     | Y       | [`triton_kernel_fused_experts`][vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts],</br>[`OAITritonExperts`][vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe.OAITritonExperts]                                                                    |
| deep gemm+triton<sup>2</sup> | standard,</br>batched | all<sup>1</sup>  | G(128),A,T    | silu, gelu                                                  | <sup>6</sup>          | Y       | [`TritonOrDeepGemmExperts`][vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe.TritonOrDeepGemmExperts],</br>[`BatchedTritonOrDeepGemmExperts`][vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe.BatchedTritonOrDeepGemmExperts]                                                 |
| marlin                       | standard              | <sup>3</sup>     | <sup>3</sup>  | silu,</br>swigluoai                                         | Y                     | Y       | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.BatchedMarlinExperts]          |
| marlin experts               | standard,</br>batched | N/A              | N/A           | silu,</br>swigluoai                                         | Y                     | Y       | [`MarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.BatchedMarlinExperts]                                                                                                            |
| trtllm                       | standard              | mxfp4,</br>nvfp4 | G(16),G(32)   | <sup>5</sup>                                                | N                     | Y       | [`TrtLlmGenExperts`][vllm.model_executor.layers.fused_moe.trtllm_moe.TrtLlmGenExperts]                                                                                                                                                                                                                      |
| pallas                       | standard              | N/A              | N/A           | silu                                                        | N                     | N       | [`fused_moe`][vllm.model_executor.layers.fused_moe.moe_pallas.fused_moe]                                                                                                                                                                                                                                    |
| iterative                    | standard              | N/A              | N/A           | silu                                                        | N                     | N       | [`fused_moe`][vllm.model_executor.layers.fused_moe.moe_torch_iterative.fused_moe]                                                                                                                                                                                                                           |
| rocm aiter moe               | standard              | fp8              | G(128),A,T    | silu, gelu                                                  | Y                     | N       | [`rocm_aiter_fused_experts`][vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_moe_impl]                                                                                                                                                                                           |
| cpu_fused_moe                | standard              | N/A              | N/A           | silu                                                        | N                     | N       | [`CPUFusedMOE`][vllm.model_executor.layers.fused_moe.cpu_fused_moe.CPUFusedMOE]                                                                                                                                                                                                                             |
| naive batched<sup>4</sup>    | batched               | int8,</br>fp8    | G,A,T         | silu, gelu                                                  | <sup>6</sup>          | Y       | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.fused_batched_moe.NaiveBatchedExperts]                                                                                                                                                                                                         |

!!! info "表格注释"
    1. 支持所有类型：mxfp4、nvfp4、int4、int8、fp8
    2. 一个将 triton 和 deep gemm experts 包装在一起的调度器，会根据类型、形状和量化参数自动选择
    3. 支持 uint4、uint8、fp8、fp4
    4. 这是一个简单的 batched 格式 experts 实现，主要用于测试
    5. `activation`