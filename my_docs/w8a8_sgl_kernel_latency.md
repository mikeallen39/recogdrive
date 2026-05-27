# W8A8 SGL Kernel Latency 对比记录

本文档记录 RecogDrive 2B 在当前优化配置下，BF16 baseline 与 W8A8 `sgl_kernel.int8_scaled_mm` 真量化路径的端到端 latency 对比。

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
| W8A8 SGL kernel + fused activation quant | 91.465 ms | 52.124 ms | 16.420 ms | 29.148 ms | 38.174 ms |
| W8A8 fused quant - BF16 | +3.204 ms | +3.324 ms | -0.650 ms | +3.557 ms | +0.013 ms |

结论：

- 当前 W8A8 SGL kernel 路径比 BF16 baseline 慢。
- 端到端 GPU latency 从 `88.261 ms` 增加到 `130.174 ms`，增加约 `41.9 ms`。
- 主要变慢来自 VLM，VLM 从 `48.800 ms` 增加到 `89.001 ms`。
- Diffusion 部分基本不是差异来源，两次测试中 diffusion 分别为 `38.161 ms` 和 `39.961 ms`。
- 加入 fused activation quantization 后，端到端 GPU latency 降到 `91.465 ms`，已经接近 BF16 baseline。
- fused activation quantization 后，VLM latency 从 `89.001 ms` 降到 `52.124 ms`，说明动态 activation quantization 的分步 kernel launch / memory pass 是之前 W8A8 变慢的主要原因。

fused activation quantization 结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_sgl_kernel_fused_quant_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

单独 kernel smoke test：

```text
shape = [1024, 2048], dtype = bf16
scale max err = 0.0
q max diff = 1
q mean diff = 0.000157
fused_quant_ms = 0.0112 ms
torch_quant_ms = 0.0881 ms
```

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

## Linear Microbenchmark

为了判断 W8A8 从 BF16 切换到 int8 后，单独 Linear 计算本身是否有收益，在 GPU6 上做了 microbenchmark。该测试拆分为三项：

- `BF16 Linear`：直接执行 `F.linear(x, weight, bias)`。
- `Int8 GEMM only`：输入 activation 和 weight 均已经预先量化，只测 `sgl_kernel.int8_scaled_mm`。
- `W8A8 total`：在线执行 fused activation quantization，再执行 `sgl_kernel.int8_scaled_mm`。

测试结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_linear_microbench_gpu6.json
```

| Shape | BF16 Linear | Int8 GEMM only | Fused quant | W8A8 total | Int8 GEMM speedup | W8A8 total speedup |
|---|---:|---:|---:|---:|---:|---:|
| LLM attn/o `M=580,K=1536,N=1536` | 0.0266 ms | 0.0174 ms | 0.0086 ms | 0.0243 ms | 1.52x | 1.09x |
| LLM FFN up/gate `M=580,K=1536,N=8960` | 0.0920 ms | 0.0551 ms | 0.0087 ms | 0.0654 ms | 1.67x | 1.41x |
| LLM FFN down `M=580,K=8960,N=1536` | 0.1009 ms | 0.0658 ms | 0.0223 ms | 0.1009 ms | 1.53x | 1.00x |
| Vision attn/o `M=2304,K=1024,N=1024` | 0.0343 ms | 0.0236 ms | 0.0128 ms | 0.0360 ms | 1.45x | 0.95x |
| Vision FFN up `M=2304,K=1024,N=4096` | 0.0982 ms | 0.0614 ms | 0.0126 ms | 0.0794 ms | 1.60x | 1.24x |
| Vision FFN down `M=2304,K=4096,N=1024` | 0.1101 ms | 0.0614 ms | 0.0417 ms | 0.1108 ms | 1.79x | 0.99x |

结论：

- 单看 `int8_scaled_mm`，W8A8 的 Linear 计算本身是有收益的，速度约为 BF16 Linear 的 `1.45x` 到 `1.79x`。
- 加上 activation quantization 后，收益明显收窄。
- FFN up/gate projection 最值得量化，例如 LLM `1536 -> 8960` 的 W8A8 total 仍有 `1.41x`。
- down projection 的 `K` 很大，activation quantization 需要扫描更长的 token hidden 维度，量化开销会吃掉 int8 GEMM 的收益。
- attention / o projection 的收益较小或不稳定，尤其 vision attention/o 在 total latency 下略慢于 BF16。

### 为什么 Int8 GEMM Only 没有达到 BF16 Linear 的 2 倍速度

理论上 int8 Tensor Core 的峰值吞吐通常高于 BF16，容易让人预期能接近 `2x`。但当前 microbenchmark 中 `int8_scaled_mm` 只有 `1.45x` 到 `1.79x`，原因主要有以下几点。

第一，当前测的不是普通 int8 GEMM，而是 `scaled_mm`。它不只是做 `int8 x int8 -> int32` 矩阵乘，还要在输出侧应用 activation scale 和 weight scale，并输出 BF16。这部分 scale 处理和 epilogue 会额外消耗带宽和指令，不能按纯 int8 GEMM 峰值估算。

第二，BF16 baseline 已经很强。A800 上 BF16 Tensor Core、cuBLAS / PyTorch Linear 路径非常成熟，对这些规则矩阵 shape 的利用率很高。W8A8 不是和低效 FP32 比，而是在和高度优化的 BF16 GEMM 比。

第三，实际 shape 不一定能把 int8 Tensor Core 吃满。当前典型 shape 是 `M=580` 或 `M=2304`，`K/N` 由 InternVL 2B 的 hidden / FFN 维度决定。部分 shape 的 tile 利用率、wave quantization、occupancy、memory access pattern 不一定正好落在 SGL kernel 的最优区间。

第四，`sgl_kernel.int8_scaled_mm` 是通用 kernel，并不是专门为 RecogDrive / InternVL 2B 的固定 shape autotune 出来的。BF16 Linear 往往走 cuBLASLt 的成熟 heuristic，而当前 int8 kernel 的 tile 选择未必对每个 projection 最优。

第五，当前 benchmark 保留了 bias 和 BF16 输出，实际输出仍要写回 BF16 tensor。对于这些中小矩阵，输出写回、scale 读写、bias 加法等非 GEMM 主体开销占比不可忽略，会降低相对于 BF16 的理论加速比。

因此，当前更合理的优化策略不是假设所有 Linear 都能获得 `2x`，而是按 shape 做选择性量化：

- 优先量化 FFN up/gate 这种 `K` 适中、`N` 很大的 projection。
- 谨慎量化 down projection，因为 activation quantization 成本随 `K` 增大明显增加。
- attention q/k/v/o projection 需要结合端到端 profiling 判断，不能默认量化。

## LLM Up/Gate 选择性量化

基于 microbenchmark，进一步测试只量化 Qwen2 LLM MLP 中的 `gate_proj` 和 `up_proj`，保留 attention、`down_proj`、vision encoder 为 BF16。

### Naive Linear 替换

第一版实现直接把 `gate_proj` 和 `up_proj` 两个 Linear 分别替换成 `W8A8Int8Linear`。该实现的问题是 Qwen2 MLP 中两个 projection 共享同一个输入 `x`，但 naive 替换会重复做两次 activation quantization：

```text
gate_proj: quant(x) + int8 GEMM
up_proj:   quant(x) + int8 GEMM
```

测试结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_llm_up_gate_only_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

同 GPU6 BF16 baseline 对比：

| 配置 | E2E mean | VLM mean | LLM mean | Diffusion mean |
|---|---:|---:|---:|---:|
| BF16 current GPU6 | 89.789 ms | 49.610 ms | 25.880 ms | 38.360 ms |
| Up/Gate naive Linear W8A8 | 89.095 ms | 50.105 ms | 26.967 ms | 37.230 ms |
| Delta | -0.695 ms | +0.495 ms | +1.087 ms | -1.129 ms |

结论：naive Linear 替换没有带来真实 VLM 加速。E2E mean 略低主要来自 diffusion 波动，而 VLM / LLM 本身变慢。

### MLP Wrapper 共享 Quantization

第二版实现把整个 Qwen2 MLP 替换为 wrapper：

```text
x_q, x_scale = quant(x)
gate = int8_scaled_mm(x_q, gate_weight, x_scale, gate_scale)
up   = int8_scaled_mm(x_q, up_weight,   x_scale, up_scale)
out  = down_proj(act(gate) * up)
```

这样 `gate_proj` 和 `up_proj` 共享同一次 activation quantization，`down_proj` 保持 BF16。

测试结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_llm_up_gate_shared_quant_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

同 GPU6 BF16 baseline 对比：

| 配置 | E2E mean | VLM mean | LLM mean | Diffusion mean |
|---|---:|---:|---:|---:|
| BF16 current GPU6 | 89.789 ms | 49.610 ms | 25.880 ms | 38.360 ms |
| Up/Gate MLP shared-quant W8A8 | 88.933 ms | 49.140 ms | 25.574 ms | 37.368 ms |
| Delta | -0.856 ms | -0.470 ms | -0.306 ms | -0.991 ms |

结论：

- 共享 activation quantization 后，LLM 从 `+1.087 ms` 变成 `-0.306 ms`，说明重复 quantization 是 naive up/gate 量化变慢的主要原因。
- VLM mean 从 `49.610 ms` 降到 `49.140 ms`，获得约 `0.47 ms` 的真实 VLM 加速。
- E2E mean 降低约 `0.86 ms`，其中约 `0.47 ms` 来自 VLM，剩余主要来自本次 diffusion 测量波动。
- 当前选择性 W8A8 的收益存在但较小，说明进一步优化需要继续减少 kernel launch，或者在更多真正有收益的模块上做共享 quantization / grouped GEMM。

## 降低 Activation 动态量化开销的方向

当前 W8A8 真量化中，weight 是初始化时静态量化并缓存为 int8 buffer，forward 阶段的主要额外开销来自 activation 动态量化。虽然 fused activation quantization 已经把 `amax + scale + round/clamp + cast int8` 合成单个 CUDA kernel，但端到端结果显示 activation quantization 仍会明显吃掉 int8 GEMM 的收益，尤其是 down projection 这类 `K` 较大的 Linear。

后续优化应优先减少 activation quantization 的次数，而不是继续单独优化单次 quant kernel。

### 1. 共享同输入的 Activation Quantization

同一个 hidden state 同时输入多个 Linear 时，应只做一次 activation quantization，然后复用 `x_q / x_scale`。

已验证的例子是 Qwen2 MLP 的 `gate_proj` 和 `up_proj`。naive Linear 替换会对同一个 `x` 重复量化两次；改成 MLP wrapper 后只量化一次，LLM latency 从比 BF16 慢 `+1.087 ms` 变成比 BF16 快 `-0.306 ms`。

可继续尝试的方向：

- 对 attention 的 `q_proj / k_proj / v_proj` 做 QKV wrapper，共享一次 activation quantization。
- 将 QKV weight 在输出维度 concat，尽量用一次 int8 GEMM 产生 `qkv`，再 split。
- 对 MLP `gate_proj / up_proj` 进一步 concat weight，用一次 int8 GEMM 代替当前两次 `int8_scaled_mm`。

该方向风险较低，因为量化粒度和数值路径不变，主要改变 kernel launch 数量和 GEMM 组织方式。

### 2. Static Activation Scale / Calibration

可以用一批 navtest 或训练集样本离线统计每层 activation scale，推理时不再做 per-token `amax` reduction，只执行 `x / scale -> int8`。

潜在收益：

- 去掉动态 `amax` reduction。
- 减少 per-token scale 计算和 scale tensor 写回。
- 对固定场景、固定 prompt、固定图像 tile 数的 RecogDrive 推理比较友好。

主要风险是精度下降。RecogDrive 的场景分布、图像 token 和语言 token 的 activation 范围可能有明显长尾，直接使用静态 scale 可能导致局部饱和。建议先只在 LLM FFN `gate/up` 上尝试，再扩展到 attention 或 vision encoder。

### 3. SmoothQuant 类方法

SmoothQuant 的思路是把 activation outlier 的一部分缩放迁移到 weight 上，使 activation 分布更平滑，从而更适合静态或低成本 activation quantization。

相比直接 static scale，它的精度风险更低，但需要 calibration，并且需要修改 weight：

```text
Y = X W
X_smooth = X / s
W_smooth = s W
Y = X_smooth W_smooth
```

可行实验路径：

- 对 LLM Linear 做 per-channel smoothing calibration。
- 优先覆盖 `gate_proj / up_proj`，因为这两个 projection 当前最有 W8A8 收益。
- 保留 `down_proj` BF16，避免大 `K` 下 activation quantization 成本和精度风险同时放大。

该方向更接近实际部署方案，但实现复杂度高于共享 quant wrapper。

### 4. Producer + Quant Fusion

很多 int8 Linear 的输入并不是原始 tensor，而是前一个轻量算子的输出，例如 RMSNorm、residual add、AdaLN、SiLU/mul 等。如果先把这些结果写回 BF16，再单独读出来做 activation quantization，会产生额外 kernel launch 和显存读写。

可以考虑的融合：

- `RMSNorm + activation quant`
- `residual add + RMSNorm + activation quant`
- `SiLU(gate) * up + activation quant`

该方向可以直接减少 kernel launch 和中间 tensor 读写，但需要更深的自定义 kernel。考虑到最终目标是 910B，不建议只为 A800 写过深的 CUDA-only 实现；更适合作为验证理论上限，或后续迁移到 Ascend C / CANN 自定义算子。

### 5. 降低 Scale 粒度

当前 activation 是 per-token 动态 scale。可以尝试 per-sequence、per-block token、或者固定 group token scale，减少 reduction 数量和 scale tensor 开销。

这个方向的代价是量化更粗，精度风险高于 per-token。自动驾驶输入具有结构化特点，理论上可以按 token 类型或空间区域分组，但这会引入额外设计复杂度。建议排在共享 quantization、static scale / SmoothQuant 之后。

### 优先级建议

短期最值得做的是：

1. `gate/up` concat GEMM：在已有 MLP shared-quant 基础上进一步减少一次 GEMM launch。
2. QKV wrapper：如果 attention projection 的 microbenchmark 显示有收益，再共享一次 activation quantization 并 concat QKV weight。
3. static activation scale / SmoothQuant：用于验证能否从根本上减少动态 quantization 成本。

不建议优先做的是通用 dynamic activation quant + GEMM 单 kernel 深度融合。开源生态中成熟可直接复用的实现较少，而且最终部署目标是 910B，CUDA-only 的深度融合维护成本和迁移成本都偏高。

## 共享 Activation Quantization 与 Producer+Quant 初步实验

本轮实验基于以下配置：

```text
2B + uniform pruning 0.50 + DDIM3 + image_max_num=6 / 3 tiles
+ pil_parallel_no_resize + addcmul_pointwise + fast DDIM + compact_v1
```

测试均在 GPU2 上使用 CUDA event，`num_samples=50`，`warmup=5`。

### Gate/Up Concat GEMM

在已有 `W8A8Qwen2UpGateMLP` 的基础上新增 `w8a8_int8_up_gate_concat` 模式。旧实现已经共享一次 activation quantization，但仍分别执行两次 GEMM：

```text
x_q, x_scale = quant(x)
gate = int8_scaled_mm(x_q, gate_weight)
up   = int8_scaled_mm(x_q, up_weight)
```

新实现将 `gate_proj` 和 `up_proj` 的 int8 weight / scale 在输出维度 concat，改成一次 GEMM 后再 split：

```text
x_q, x_scale = quant(x)
gate_up = int8_scaled_mm(x_q, cat([gate_weight, up_weight]))
gate, up = split(gate_up)
```

数值等价性：dummy MLP smoke test 中，旧 shared-quant 与 concat GEMM 输出最大绝对误差为 `0.0`。

纯 GEMM microbenchmark，shape 为 `M=580,K=1536,N_each=8960`：

| 配置 | Latency |
|---|---:|
| separate gate/up GEMM only | 0.1179 ms |
| concat gate/up GEMM only | 0.1038 ms |
| separate gate/up + shared quant | 0.1306 ms |
| concat gate/up + shared quant | 0.1120 ms |

microbenchmark 显示每层 gate/up concat 大约节省 `0.0186 ms`。但端到端 LLM 统计中收益明显被整体 forward 噪声和其它模块开销稀释。

### QKV Shared Quant / Concat GEMM

新增 `w8a8_int8_up_gate_concat_qkv` 模式。在 `gate/up concat` 基础上，额外将 Qwen2 attention 的 `q_proj / k_proj / v_proj` 替换为 QKV wrapper：

```text
x_q, x_scale = quant(hidden_states)
qkv = int8_scaled_mm(x_q, cat([q_weight, k_weight, v_weight]))
q, k, v = split(qkv)
```

`o_proj` 保持 BF16，因为此前 microbenchmark 显示 attention/o projection 的 W8A8 total 收益较小，直接量化风险更高。

### End-to-End Latency

结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/bf16_gpu2_rerun_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_llm_up_gate_shared_quant_gpu2_rerun_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_llm_up_gate_concat_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_llm_up_gate_concat_qkv_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

| 配置 | E2E mean | VLM mean | LLM mean | Vision mean | Diffusion mean |
|---|---:|---:|---:|---:|---:|
| BF16 GPU2 rerun | 88.429 ms | 49.283 ms | 25.548 ms | 17.086 ms | 38.083 ms |
| Up/Gate shared-quant W8A8 | 87.391 ms | 49.268 ms | 25.589 ms | 17.078 ms | 37.077 ms |
| Up/Gate concat W8A8 | 87.612 ms | 49.288 ms | 25.502 ms | 17.097 ms | 37.077 ms |
| Up/Gate concat + QKV W8A8 | 84.598 ms | 48.175 ms | 24.450 ms | 17.002 ms | 35.991 ms |

对比 BF16 GPU2 rerun：

- `Up/Gate shared-quant` 对 VLM/LLM 基本没有稳定收益，E2E 降低主要来自 diffusion 测量波动。
- `Up/Gate concat` 只让 LLM mean 降低约 `0.046 ms`，端到端收益不稳定；说明单独减少 gate/up 的一个 GEMM launch 不足以明显改变整体 VLM latency。
- `Up/Gate concat + QKV` 让 VLM mean 降低 `1.107 ms`，LLM mean 降低 `1.098 ms`，这是目前 W8A8 选择性量化中更有意义的收益。
- Diffusion latency 在这些实验中也有波动，但 QKV 量化并不改变 diffusion 模块结构，因此判断 VLM 优化时应优先看 `VLM mean` 和 `LLM mean`。

### RMSNorm + Quant Microbenchmark

新增 `scripts/evaluation/benchmark_rmsnorm_quant_cuda_event.py`，用于评估 producer + quant fusion 的理论收益。该实验只做 microbenchmark，尚未接入端到端模型。

结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/rmsnorm_quant_fusion_microbench_gpu2.json
```

| Shape | RMSNorm only | RMSNorm then quant | Fused RMSNorm+quant | q max diff | scale max diff |
|---|---:|---:|---:|---:|---:|
| LLM hidden `580x1536` | 0.0758 ms | 0.0871 ms | 0.0103 ms | 2 | 2.07e-4 |
| Vision hidden `2304x1024` | 0.0872 ms | 0.1000 ms | 0.0195 ms | 1 | 2.41e-4 |
| Diffusion hidden `64x1024` | 0.0752 ms | 0.0865 ms | 0.0101 ms | 1 | 1.46e-4 |

结论：

- 单独看 kernel，`RMSNorm + activation quant` 融合有明显潜力，可以把一次 producer + quant 从约 `0.087-0.100 ms` 降到 `0.010-0.019 ms`。
- 当前 fused kernel 的数值路径与 PyTorch RMSNorm 后再 quant 不完全一致，int8 输出最大差异为 `1-2`，scale 最大差异约 `1e-4`。该误差不大，但接入端到端前仍需要做 PDMS 验证。
- 下一步如果继续做 producer+quant fusion，建议先接入 LLM RMSNorm 后进入 QKV / MLP 的位置，并保留开关，避免和 QKV 量化本身的精度影响混在一起。

### 为什么 Dynamic Quant 的 Amax Reduction 开销明显

W8A8 dynamic activation quantization 的核心代价不只是 `round + cast int8`，而是每个 token 都要沿 hidden 维度做一次全量 `amax` reduction：

```text
scale[i] = max(abs(x[i, :])) / 127
```

以 LLM hidden `M=580,K=1536` 为例，per-token quantization 至少要完成以下步骤：

1. 读取每个 token 的完整 hidden 向量。
2. 对所有元素做 `abs`。
3. 在线程块内做 max reduction，并同步得到该 token 的 scale。
4. 再遍历一次输入，用 `x / scale` 做 `round + clamp + cast int8`。
5. 写出完整 int8 activation 和 per-token scale。

这类操作贵的原因：

- 它必须扫描完整输入，否则 scale 不准，容易发生 int8 饱和。
- reduction 有线程同步和规约开销，不像逐元素乘加那样完全并行。
- 即使融合成单个 kernel，也通常需要两阶段处理：先得到 scale，再量化，因此很难避免至少一次完整读输入和一次写 int8。
- 它主要是 memory/reduction bound，Tensor Core 基本帮不上忙。
- 开销近似随 `M*K` 增长，因此 `down_proj` 这类大 `K` 输入特别不划算。

这解释了为什么某些 Linear 单看 int8 GEMM 有收益，但加上 dynamic quant 后收益被吃掉：

```text
W8A8 total = dynamic activation quant + int8_scaled_mm + scale epilogue
BF16 total = highly optimized BF16 GEMM
```

当 `dynamic activation quant` 接近或超过 `BF16 GEMM - int8 GEMM` 的差值时，W8A8 total 就不会比 BF16 快。后续优化 dynamic quant 的重点应放在减少 reduction 次数、复用 quant 结果、或者用 static scale / SmoothQuant 去掉在线 amax，而不是单纯继续优化 `round/cast`。

### RMSNorm+Quant 接入端到端

新增 `w8a8_int8_rmsnorm_up_gate_qkv` 模式，把 Qwen2 decoder layer 中两处 RMSNorm producer 直接和 activation quantization 融合：

```text
input_layernorm(hidden_states) -> quant -> QKV int8 GEMM
post_attention_layernorm(hidden_states) -> quant -> gate/up int8 GEMM
```

该实现通过 decoder layer wrapper 保持原始 residual、attention、MLP 调用结构不变，只改变进入 QKV 和 gate/up int8 GEMM 前的 RMSNorm+Quant 路径。由于 RMSNorm+Quant fused kernel 的数值路径与 PyTorch RMSNorm 后再 quant 有微小差异，PDMS 需要单独验证。

Latency 结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_rmsnorm_up_gate_qkv_gpu6_rerun_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

同配置对比：

| 配置 | E2E mean | VLM mean | LLM mean | Vision mean | Diffusion mean |
|---|---:|---:|---:|---:|---:|
| BF16 GPU2 rerun | 88.429 ms | 49.283 ms | 25.548 ms | 17.086 ms | 38.083 ms |
| Up/Gate concat + QKV W8A8 | 84.598 ms | 48.175 ms | 24.450 ms | 17.002 ms | 35.991 ms |
| RMSNorm+Quant + Up/Gate concat + QKV W8A8 | 78.295 ms | 40.763 ms | 17.417 ms | 16.947 ms | 37.029 ms |

结论：

- RMSNorm+Quant 接入端到端后，LLM mean 从 `24.450 ms` 降到 `17.417 ms`，比 QKV/shared-quant 版本进一步降低约 `7.0 ms`。
- 相比 BF16，LLM mean 降低约 `8.1 ms`，VLM mean 降低约 `8.5 ms`。
- 这说明此前 dynamic quant 的主要开销并不只是 int8 cast，而是 RMSNorm producer 写回、activation 重新读取、per-token amax reduction 与 quant kernel launch 的组合开销。
- 当前最重要的风险是精度。PDMS 已启动重新验证，实验名为 `recogdrive_agent_eval_2b_uniform050_ddim3_maxnum6_pil_compact_v1_fastddim_w8a8_rmsnorm_qkv_fixed_zxz`。

### Static Activation Scale 初步测试

新增 `w8a8_int8_rmsnorm_static_up_gate_qkv` 模式，用固定 activation scale 替代 per-token dynamic scale。当前只是初步验证去掉在线 `amax` 的 latency 上限，默认：

```text
RECOGDRIVE_RMSNORM_STATIC_ACT_SCALE=0.03
```

结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_rmsnorm_static003_up_gate_qkv_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

同 GPU6 对比：

| 配置 | E2E mean | VLM mean | LLM mean | Vision mean | Diffusion mean |
|---|---:|---:|---:|---:|---:|
| RMSNorm+dynamic quant | 78.295 ms | 40.763 ms | 17.417 ms | 16.947 ms | 37.029 ms |
| RMSNorm+static scale 0.03 | 80.153 ms | 41.264 ms | 17.599 ms | 16.973 ms | 37.736 ms |

结论：

- 当前 naive static scale 没有带来 latency 收益，VLM mean 反而慢约 `0.50 ms`，LLM mean 慢约 `0.18 ms`。
- 原因是 static scale 只去掉了 quant 阶段的 `amax`，但 RMSNorm 本身仍然需要对 hidden 维度做平方和 reduction；因此整体 producer+quant 路径仍然是 reduction-bound。
- 当前实现还需要给 SGL `int8_scaled_mm` 提供 activation scale tensor，即使 scale 是常数，也仍然要写出 per-token scale buffer。
- 未做 calibration 的 static scale 精度风险较大，因此暂时不跑 PDMS。后续若继续探索 static scale，应先做 calibration / SmoothQuant，而不是手工设一个全局常数 scale。

补充 kernel microbenchmark：

| Shape | RMSNorm+dynamic quant | RMSNorm+static scale 0.03 |
|---|---:|---:|
| LLM hidden `580x1536` | 0.0103 ms | 0.0104 ms |
| Vision hidden `2304x1024` | 0.0194 ms | 0.0131 ms |
| Diffusion hidden `64x1024` | 0.0100 ms | 0.0101 ms |

LLM 形状下 static scale kernel 与 dynamic fused kernel 基本持平，因此端到端没有进一步收益是符合预期的。static scale 更可能带来收益的前提是进一步减少 scale tensor 写回、或者让后续 GEMM kernel 原生支持 scalar/per-layer scale，而不是仍然走 per-token scale tensor 接口。

### Vision Encoder LayerNorm+Quant 初步测试

背景：ReCogDrive 2B 的 InternVL vision encoder 使用 `LayerNorm`，不是 LLM 里的 `RMSNorm`：

```text
hidden -> norm1 -> attn.qkv
hidden -> norm2 -> mlp.fc1
```

因此不能直接复用 `RMSNorm+Quant` kernel。本次新增 `w8a8_int8_layernorm_qkv_fc1` vision quant mode：

```text
LayerNorm(norm1) + dynamic activation quant -> attn.qkv int8 GEMM
LayerNorm(norm2) + dynamic activation quant -> mlp.fc1 int8 GEMM
attn.proj / mlp.fc2 -> 继续使用已有 W8A8 int8 Linear
Conv2d patch embedding -> 保持 BF16
```

实现细节：

- 新增 CUDA extension 接口 `layernorm_quantize_activation_per_token_int8`。
- extension 名称从 `recogdrive_int8_quant_v2` 改为 `recogdrive_int8_quant_v3`，避免加载旧缓存 `.so`。
- vision encoder 每层 wrapper 为 `W8A8InternVisionEncoderLayerLayerNormQuant`，只替换 24 个 InternViT encoder layer，不改 patch embedding 和 position embedding。
- 小 shape 数值自检中，fused LayerNorm+Quant 与 PyTorch `layer_norm + per-token quant` 的 int8 最大差值为 `1`，主要来自 BF16/FP32 和四舍五入边界差异。

Latency 结果文件：

```text
/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/w8a8_int8_rmsnorm_up_gate_qkv_vision_layernorm_qkv_fc1_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_compact_v1_50.json
```

同配置均为：

```text
2B + uniform pruning 0.50 + DDIM3 + image_max_num=6 / 3 tiles
+ pil_parallel_no_resize + addcmul_pointwise + fast DDIM + compact_v1
+ LLM w8a8_int8_rmsnorm_up_gate_qkv
```

| 配置 | E2E mean | VLM mean | Vision mean | LLM mean | Diffusion mean |
|---|---:|---:|---:|---:|---:|
| Vision BF16 | 78.295 ms | 40.763 ms | 16.947 ms | 17.417 ms | 37.029 ms |
| Vision LayerNorm+Quant W8A8 | 79.795 ms | 40.115 ms | 16.205 ms | 17.735 ms | 38.034 ms |

结论：

- vision encoder 局部从 `16.947 ms` 降到 `16.205 ms`，收益约 `0.74 ms`。
- VLM mean 从 `40.763 ms` 降到 `40.115 ms`，收益约 `0.65 ms`。
- E2E mean 没有改善，主要是本次 50-sample 中 diffusion 和 E2E 计时有少量抖动，且 vision 局部收益本身较小。
- 该方案说明 `LayerNorm producer + quant` 在 vision encoder 中可用，但收益明显小于 LLM 的 RMSNorm+Quant。后续如果继续优化 vision encoder，优先级应高于这个小融合的是 token 数缩减、ViT 内部 early token pruning/merge，或者更深的 attention/MLP 结构级优化。
