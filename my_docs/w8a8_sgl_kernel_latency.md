# W8A8 SGL Kernel Latency 对比记录

本文档记录 RecogDrive 2B 在当前优化配置下，BF16 baseline 与 W8A8 `sgl_kernel.int8_scaled_mm` 真量化路径的端到端 latency 对比。`torch._int_mm` 结果不作为有效对比对象，因为该路径没有使用 fused scaled int8 GEMM，前后处理开销过大，不能代表可用的 W8A8 实现。

## 测试配置

共同配置：

- 模型：ReCogDrive 2B
- 数据集：navtest
- GPU：NVIDIA A800 80GB PCIe
- CUDA event latency samples：warmup 5，正式统计 50 samples
- Visual token pruning：uniform pruning，keep ratio `0.50`
- Diffusion：DDIM，`3` steps
- 图像输入：`image_max_num=6`，实际 `3 tiles`
- 图像 backend：`pil_parallel_no_resize`
- Diffusion pointwise：`addcmul_pointwise`
- Action head：`fast_ddim_action=True`
- Prompt：`compact_v1`
- FA2：开启

BF16 baseline 结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/prompt_compact_v1_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_50.json
```

W8A8 SGL kernel 结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_sgl_kernel_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

## 总体结果

| 配置 | E2E GPU mean | VLM mean | Vision encoder mean | LLM mean | Diffusion mean |
|---|---:|---:|---:|---:|---:|
| BF16 baseline | 88.261 ms | 48.800 ms | 17.070 ms | 25.591 ms | 38.161 ms |
| W8A8 SGL kernel | 130.174 ms | 89.001 ms | 32.227 ms | 48.319 ms | 39.961 ms |
| W8A8 - BF16 | +41.913 ms | +40.201 ms | +15.157 ms | +22.728 ms | +1.800 ms |

结论：

- 当前 W8A8 SGL kernel 路径比 BF16 baseline 慢。
- 端到端 GPU latency 从 `88.261 ms` 增加到 `130.174 ms`，增加约 `41.9 ms`。
- 主要变慢来自 VLM，VLM 从 `48.800 ms` 增加到 `89.001 ms`。
- Diffusion 部分基本不是差异来源，两次测试中 diffusion 分别为 `38.161 ms` 和 `39.961 ms`。

## 细分观察

### Vision encoder

BF16 baseline：

```text
vision_encoder_cuda_ms mean = 17.070 ms
```

W8A8 SGL kernel：

```text
vision_encoder_cuda_ms mean = 32.227 ms
```

增加：

```text
+15.157 ms
```

当前 vision encoder 中替换了 `96` 个 Linear，但 patch embedding Conv2d 仍保持 BF16。结果说明当前 W8A8 动态量化路径不适合直接套到 vision encoder 所有 Linear 上：vision encoder 的矩阵规模和 batch/token 形状并不一定足够大，BF16 GEMM 本身已经很快，而每层额外的动态 activation quantization 开销会显著增加 latency。

### LLM

BF16 baseline：

```text
language_model_cuda_ms mean = 25.591 ms
```

W8A8 SGL kernel：

```text
language_model_cuda_ms mean = 48.319 ms
```

增加：

```text
+22.728 ms
```

当前 LLM 中替换了 `196` 个 Linear。虽然 `sgl_kernel.int8_scaled_mm` 融合了 int8 GEMM 与 scale 处理，但每个 Linear 前仍然需要动态 activation quantization，包括 per-token `amax`、scale 计算、除法、round、clamp、cast int8 和必要的 contiguous/layout 处理。这些额外 kernel launch 和 memory pass 在小 batch / 中等 seq len 场景下很容易超过 int8 GEMM 本身节省的时间。

## 当前 W8A8 实现为什么仍然慢

当前 `W8A8Int8Linear` 的核心流程是：

1. 将输入 reshape 成 `[num_tokens, in_features]`。
2. 对 activation 做 per-token 动态 int8 量化。
3. 调用 `sgl_kernel.int8_scaled_mm(x_q, qweight.t(), x_scale, weight_scale, out_dtype, bias)`。
4. reshape 回原始输出形状。

其中第 3 步使用了 fused scaled int8 GEMM，是正确方向。但第 2 步仍然是逐层动态量化，且没有与 GEMM 融合。对于每个 Linear，都会额外产生：

- activation 转 float / 读写一遍；
- per-token `amax` reduction；
- scale 计算；
- activation 除以 scale；
- round；
- clamp；
- cast 到 int8；
- contiguous/layout 处理。

RecogDrive 当前 VLM 推理是小 batch 场景，视觉 token 经过 pruning 后约为 `384`，输入序列长度约为 `580` 左右。这个规模下，BF16 Linear 在 A800 上已经非常高效；如果 W8A8 不能把 activation quantization 和 GEMM 前后处理进一步融合，就很难比 BF16 更快。

## 初步结论

当前 W8A8 SGL kernel 实验说明：

- `sgl_kernel.int8_scaled_mm` 可以成功接入 RecogDrive，并且比普通 `torch._int_mm` 路径合理得多。
- 但“直接把所有 Linear 替换成动态 W8A8”并不能带来端到端加速。
- 当前瓶颈不是 diffusion，而是 VLM 中大量 Linear 的动态 activation quantization 开销。
- 后续优化不应该继续把全部模块无差别量化，而应该做选择性量化和量化前后处理融合。

## 后续优化方向

优先建议做以下 ablation：

1. 只量化 LLM，不量化 vision encoder。
2. 只量化 LLM FFN 大矩阵，例如 gate / up / down projection，跳过 attention qkv/o projection。
3. 只量化大 Linear，按 `in_features * out_features` 或实际 profiling 过滤小矩阵。
4. 尝试 W8A16 / weight-only 量化，避免动态 activation quantization。
5. 尝试 fused activation quantization kernel，至少把 `amax + scale + quantize` 融合成单个 CUDA kernel。
6. 如果后续目标仍是 910B，需要优先考虑 Ascend 上可用的 int8 matmul / quantize 融合能力，而不是只针对 CUDA kernel 做过深优化。

