# ReCogDrive 2B/8B 推理延迟分析

测试时间：2026-05-18 UTC

测试机器：NVIDIA A800 80GB PCIe

测试环境：`/data/zxz/condaenv/curious_vla/navsim`

测试脚本：`scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py`

测试数据：NAVSIM navtest 样本，`warmup=5`，有效统计样本数 `50`

计时方法：GPU 计算段使用 `torch.cuda.Event(enable_timing=True)`，CPU 数据处理段使用 wall-clock 作为参考。

## 结果文件

- 2B-RL，FlashAttention2：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_cuda_event_50_fa2.json`
- 8B-RL，FlashAttention2：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_8b_cuda_event_50_fa2.json`
- 2B-RL，FA2 + uniform token pruning 10%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_prune_uniform_0.10_50.json`
- 2B-RL，FA2 + T-FPS token pruning 10%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_prune_tfps_0.10_50.json`
- 2B-RL，FA2 + T-FPS token pruning 25%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_prune_tfps_0.25_50.json`
- 2B-RL，FA2 + T-FPS token pruning 50%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_prune_tfps_0.50_50.json`
- 2B-RL，FA2 + VLM 拆分 profiling baseline：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_profile_baseline_50.json`
- 2B-RL，FA2 + VLM 拆分 profiling T-FPS 25%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_profile_tfps_0.25_50.json`
- 2B-RL，FA2 + VLM 拆分 profiling T-FPS 50%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_profile_tfps_0.50_50.json`

FlashAttention2 复测环境：

- `torch=2.5.1+cu121`
- `flash-attn=2.7.4.post1`
- wheel：`flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`

## 总表

端到端 latency 采用同步单次请求口径：

`feature_wall_ms + image_preprocess_wall_ms + e2e_gpu_cuda_ms + postprocess_wall_ms`

| 配置 | PDMS | 端到端 latency(ms) |
|---|---:|---:|
| 2B baseline, DDIM 5 | 0.904283 | 310.210 |
| 8B baseline, DDIM 5 | 0.903133 | 471.152 |
| 2B DDIM 3 | 0.907049 | 277.699 |
| 2B DDIM 2 | 0.883279 | 261.931 |
| 2B DDIM 1 | 0.013528 | 245.402 |
| 2B uniform pruning 0.10 | 0.671761 | 250.702 |
| 2B uniform pruning 0.25 | 0.781920 | 260.373 |
| 2B uniform pruning 0.50 | 0.881033 | 272.212 |
| 2B uniform pruning 0.50 + DDIM 3 | 0.879912 | 234.602 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles | 0.850495 | 167.491 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + OpenCV image backend | 0.837151 | 133.029 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + OpenCV + addcmul pointwise fusion | 同 OpenCV 行（数值等价） | 129.471 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM | 同 PIL 行（数值等价） | 137.370 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM + compact prompt v1 | 0.849623 | 131.476 |
| 2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM + compact prompt v1 + LLM W8A8 fake quant | 0.849867 | 未测 |
| 2B T-FPS pruning 0.10 | 0.691985 | 265.084 |
| 2B T-FPS pruning 0.25 | 0.801383 | 295.150 |
| 2B T-FPS pruning 0.50 | 0.866982 | 352.433 |
| 2B uniform merging 0.25 | 0.737774 | 252.952 |
| 2B uniform merging 0.50 | 0.872196 | 270.941 |

## 原始模型 Navtest 精度

测试设置：

- 数据集：NAVSIM navtest，完整 `12146` 个 scenario。
- 模型：ReCogDrive 2B-RL / 8B-RL，未做视觉 token pruning。
- 运行结果：两个模型均 `12146/12146` 成功，`0` failed。

结果文件：

- 2B-RL：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_rl_zxz/2026.05.17.09.51.16/2026.05.17.12.20.57.csv`
- 8B-RL：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_8b_rl_zxz/2026.05.17.12.22.30/2026.05.17.15.45.25.csv`

| 模型 | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B-RL | 0.904283 | 0.981023 | 0.976288 | 0.858269 | 0.950601 | 1.000000 | 0.966615 |
| 8B-RL | 0.903133 | 0.976741 | 0.976371 | 0.861694 | 0.945826 | 1.000000 | 0.975342 |

结论：当前复现中 2B-RL 和 8B-RL 的 navtest PDMS 非常接近，2B-RL 略高 `0.00115`。后续 pruning 实验默认以 2B-RL `0.904283` 作为 baseline。

## 测试拆分

- `image_preprocess_wall_ms`：PIL 读图、dynamic preprocess、resize、normalize、stack 的 CPU 时间。
- `image_h2d_cuda_ms`：图像 tensor 拼接并搬到 GPU 的 CUDA event 时间。
- `vlm_cuda_ms`：InternVL backbone 前向的 CUDA event 时间。
- `vision_encoder_cuda_ms`：InternVL vision encoder / `extract_feature()` 的 CUDA event 时间。
- `token_select_cuda_ms`：视觉 token selection 的 CUDA event 时间。baseline 没有 pruning，因此为 `0`。
- `language_model_cuda_ms`：InternVL language model forward 的 CUDA event 时间。
- `diffusion_cuda_ms`：diffusion planner/action head 的 CUDA event 时间。
- `e2e_gpu_cuda_ms`：从图像 tensor H2D、VLM forward 到 diffusion planner 的 GPU 端到端 CUDA event 时间。

注意：`e2e_gpu_cuda_ms` 不包含 CPU 图片预处理时间；真实在线系统如果没有异步预处理，需要额外考虑 `image_preprocess_wall_ms`。

## 单次请求完整 Latency

这里的“单次请求完整 latency”按同步在线推理口径估算：

`feature_wall_ms + image_preprocess_wall_ms + e2e_gpu_cuda_ms + postprocess_wall_ms`

该口径包含特征构造、CPU 图片预处理、图像 H2D、VLM forward、diffusion planner 和预测轨迹回 CPU；不包含数据集 dataloader 查 token、日志读取、评测打分、进程间调度等 navtest 框架开销。如果线上把 CPU 图片预处理和 GPU 推理做流水并行，实际端到端关键路径会低于该同步估算。

| 配置 | feature(ms) | 图片预处理 CPU(ms) | e2e GPU(ms) | postprocess(ms) | 单次请求完整 latency(ms) |
|---|---:|---:|---:|---:|---:|
| 2B + FA2 baseline | 0.198 | 92.291 | 237.682 | 0.039 | 330.210 |
| 8B + FA2 baseline | 0.192 | 75.323 | 395.597 | 0.040 | 471.152 |
| 2B + FA2 + uniform 10% | 0.202 | 80.513 | 169.948 | 0.039 | 250.702 |
| 2B + FA2 + T-FPS 10% | 0.202 | 78.126 | 186.716 | 0.040 | 265.084 |
| 2B + FA2 + T-FPS 25% | 0.200 | 78.820 | 216.092 | 0.039 | 295.150 |
| 2B + FA2 + T-FPS 50% | 0.193 | 75.753 | 276.448 | 0.040 | 352.433 |

结论：

- 2B + FA2 baseline 的同步完整请求 latency 约 `330 ms`，其中 CPU 图片预处理约占 `92 ms`，不可忽略。
- 8B + FA2 baseline 的同步完整请求 latency 约 `471 ms`，主要增加来自 VLM forward。
- `T-FPS@0.50` 在 GPU 端已经慢于 baseline，同步完整 latency 也更高，约 `352 ms`。
- 如果目标是实际在线端到端延迟，除了 VLM token pruning，还需要优化或异步化 CPU 图片预处理。

## CPU 图片预处理 Profiling

针对当前最优附近配置 `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles`，进一步拆分 CPU 图片预处理。原始 PIL 路径主要耗时来自 `Image.open`、JPEG decode + RGB convert、`dynamic_preprocess` 中的 resize，以及最终 normalize/stack。

关键观测：

- 原始 PIL 路径下，单次请求图片预处理约 `55.319 ms`；更早未限制 `max_num` 的 baseline 中图片预处理为 `75-95 ms` 量级。
- `dynamic_preprocess` 中有两次主要 resize：原图缩放到动态 tile 大小，以及生成 `448x448` thumbnail；其中 resize 是 CPU 侧最大瓶颈。
- `Image.open` 本身只是懒加载，真正 JPEG decode 通常在后续 `convert("RGB")` 或 resize 时触发，因此 `open / decode+convert / resize` 三者在 PIL 路径里会互相耦合。
- 改成 OpenCV backend 后，`cv2.imread + cv2.cvtColor + cv2.resize + numpy-to-torch normalize` 将图片预处理降到 `23.235 ms`。

对比结果：

| 配置 | 图片预处理 CPU(ms) | VLM(ms) | diffusion(ms) | e2e GPU(ms) | 完整 latency(ms) |
|---|---:|---:|---:|---:|---:|
| PIL backend | 55.319 | 55.578 | 54.275 | 111.935 | 167.491 |
| OpenCV backend | 23.235 | 54.779 | 53.893 | 109.591 | 133.029 |

结论：

- OpenCV backend 的主要收益来自 CPU 图片解码和 resize，GPU 侧 VLM / diffusion 基本不变。
- OpenCV 后完整 latency 从 `167.491 ms` 降到 `133.029 ms`，节省约 `34.5 ms`。
- OpenCV 后 navtest PDMS 为 `0.837151`，相比同配置 PIL backend 的 `0.850495` 下降 `0.013344`。
- 如果不能做预缓存，后续 CPU 侧可继续考虑更高效的在线 decode/resize 后端，但单靠图片预处理已经很难补足到 `100 ms` 内的全部差距。

OpenCV 精度结果：

- 结果文件：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_prune_uniform_050_ddim3_maxnum6_opencv_zxz/2026.05.23.02.15.45/2026.05.23.03.36.02.csv`
- 成功场景：`12146/12146`
- failed：`0`

| 配置 | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PIL backend | 0.850495 | - | - | - | - | - | - |
| OpenCV backend | 0.837151 | 0.951301 | 0.936605 | 0.814433 | 0.890746 | 1.000000 | 0.957393 |

精度下降拆分：

- 相比原始 2B baseline `0.904283`，当前 OpenCV 配置总下降 `0.067132`。
- 其中 `uniform@0.50 + DDIM3` 本身仍有 `0.879912`，相对 baseline 只下降 `0.024371`。
- `image max_num=6 / 3 tiles` 将 PDMS 从 `0.879912` 降到 `0.850495`，额外下降 `0.029417`，这是更主要的精度损失来源。
- OpenCV backend 再从 `0.850495` 降到 `0.837151`，额外下降 `0.013344`。

原因判断：

- `max_num=6 / 3 tiles` 会减少动态 tiling 覆盖的视野细节；虽然仍是前视单图，但 tile 数减少会影响远处小目标、车道边界和局部几何信息，因此 PDMS 会下降。
- OpenCV 路径与原 PIL/torchvision 路径不完全数值等价：JPEG decode、RGB 转换、resize interpolation、rounding、归一化执行顺序都可能产生像素级差异。VLM 对这类输入分布差异比较敏感，尤其当前已经叠加 `uniform@0.50` token pruning 和 `3 tiles` 压缩，鲁棒性余量更小。
- 因此这次“掉得挺多”不是单一 OpenCV 造成的，而是 `tile 数减少` 是主要项，`OpenCV 预处理数值差异` 是次要但可见项。

### PIL Backend 保精度优化

为避免 OpenCV 数值路径导致 PDMS 下降，进一步测试了只优化 PIL 路径的 backend：

- `pil_no_resize`：保留 `Image.open().convert("RGB")`、`dynamic_preprocess()`、crop 和 thumbnail，只去掉 tile 后 `torchvision.Resize(448,448)`。
- `pil_parallel`：保留原 transform，但并行执行 dynamic grid resize 和 thumbnail resize。
- `pil_parallel_no_resize`：同时启用 parallel resize 和 no-resize transform。

像素等价性检查：

- 抽取 20 张 navtest 前视图，对比原 `pil` 与 `pil_no_resize` / `pil_parallel` / `pil_parallel_no_resize` 的输出 tensor。
- 三个 PIL 优化 backend 均为 `max_abs_diff=0.0`、`mean_abs_diff=0.0`，`num_patches` 和 shape 完全一致。
- 因此这些 backend 理论上与原 PIL 数值等价，不需要像 OpenCV 一样额外担心 PDMS 下降。

结果文件：

- pil：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_50.json`
- pil no resize：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_no_resize_50.json`
- pil parallel：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_parallel_50.json`
- pil parallel no resize：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_parallel_no_resize_50.json`

| backend | 图片预处理 CPU mean(ms) | e2e GPU mean(ms) | 完整 latency mean(ms) | 与原 PIL tensor 等价 |
|---|---:|---:|---:|---|
| pil | 79.704 | 96.547 | 176.441 | - |
| pil_no_resize | 52.412 | 95.710 | 148.319 | 是 |
| pil_parallel | 41.220 | 96.299 | 137.708 | 是 |
| pil_parallel_no_resize | 41.395 | 94.597 | 136.188 | 是 |

结论：

- `pil_no_resize` 证明 `torchvision.Resize(448,448)` 是冗余开销，去掉后 tensor 与原 PIL 完全一致，图片预处理下降约 `27.3 ms`。
- `pil_parallel` / `pil_parallel_no_resize` 进一步把图片预处理降到约 `41 ms`，完整 latency 约 `136 ms`，接近 OpenCV 的 `133 ms`，但保持原 PIL tensor 完全等价。
- 当前最推荐的保精度图片 backend 是 `pil_parallel_no_resize`；相比 OpenCV，它牺牲约 `3 ms` latency，但避免了 OpenCV 额外 `0.013344` PDMS 下降风险。

## Prompt 压缩实验

针对 `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM`，测试了 `compact_v1` prompt。该版本只压缩 system message 和输出要求，不改历史轨迹格式、不改 high-level command 表达，因此属于低风险 prompt 压缩。

结果文件：

- full prompt latency：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/fusion_fastddim_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_50.json`
- compact prompt latency：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/prompt_compact_v1_2b_uniform050_ddim3_maxnum6_pil_parallel_no_resize_fastddim_50.json`
- compact prompt PDMS：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_uniform050_ddim3_maxnum6_pil_compact_v1_fastddim_zxz/2026.05.26.08.28.21/2026.05.26.09.32.33.csv`

| 配置 | visual tokens | seq len | VLM(ms) | LLM(ms) | diffusion(ms) | e2e GPU(ms) | 完整 latency(ms) | PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full prompt | 384 | 868.2 | 55.399 | 30.579 | 38.700 | 95.941 | 137.370 | 0.850495 |
| compact_v1 | 384 | 583.2 | 48.800 | 25.591 | 38.161 | 88.261 | 131.476 | 0.849623 |

结论：

- `compact_v1` 将非视觉 token 从约 `486` 降到约 `201`，主要收益来自 system prompt token 从约 `283` 降到约 `31`。
- VLM latency 下降 `6.599 ms`，其中 LLM 下降 `4.988 ms`；完整同步 latency 下降 `5.894 ms`。
- navtest `12146/12146` 成功、`0` failed，PDMS 为 `0.849623`，相比 full prompt 对应 PIL 配置 `0.850495` 下降 `0.000872`，目前可以认为精度基本保持。

## LLM W8A8 伪量化实验

针对 `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM + compact prompt v1`，进一步只对 VLM 中的 LLM Linear 做 W8A8 fake quant。该实现用于验证量化敏感性：权重做 per-output-channel fake quant 并缓存，激活做 per-token dynamic fake quant；matmul 仍然是 BF16/FP16 `F.linear`，不代表真实 int8 kernel 的 latency。

结果文件：

- W8A8 fake quant PDMS：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_uniform050_ddim3_maxnum6_pil_compact_v1_fastddim_w8a8_fake_llm_zxz/2026.05.26.11.58.24/2026.05.26.13.04.03.csv`
- 运行日志：`/data/zxz/HUAWEI/VLA/navsim_data/exp/logs/pdms_w8a8_fake_llm_gpu4_20260526_115813.log`

| 配置 | LLM Linear 数 | 成功场景 | failed | PDMS |
|---|---:|---:|---:|---:|
| compact_v1 baseline | - | 12146 | 0 | 0.849623 |
| compact_v1 + LLM W8A8 fake quant | 196 | 12146 | 0 | 0.849867 |

结论：

- LLM-only W8A8 fake quant 没有造成可观测精度下降；相对 compact_v1 baseline，PDMS 变化为 `+0.000244`，属于评估噪声量级。
- 当前 fake quant 不是加速实现。完整 PDMS 平均单场景耗时从约 `0.314 s/scene` 到约 `0.345 s/scene` 的量级，主要因为每个 Linear 前增加了动态激活量化的 `abs/amax/round/clamp/dequant`。
- 如果后续做真加速，需要接真实 W8A8 GEMM / 910B CANN 量化图；该实验只说明 LLM Linear 的 W8A8 数值扰动风险较低。

## JPEG Decode 与 Draft 解码

当前 NAVSIM navtest 的前视相机图像来自磁盘上的 `.jpg` 文件，模型实际输入不是 JPEG 字节流，而是解码后的 RGB tensor。当前 PIL 路径的主要流程是：

```text
JPEG 文件
-> Image.open() 读取 header / 元信息
-> convert("RGB") 触发 JPEG decode 和 RGB 转换
-> dynamic_preprocess 做动态 tile resize、crop 和 thumbnail resize
-> ToTensor + Normalize
-> torch.cat 后 H2D 到 GPU
```

注意：`Image.open()` 本身通常是 lazy load，真正 JPEG decode 往往发生在 `convert("RGB")`、`resize()` 或第一次访问像素时。因此 profiling 里应把 `open / decode+convert / resize` 作为耦合的 CPU 图片预处理路径看待。

针对 `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + PIL optimized + addcmul pointwise fusion + fast DDIM + compact prompt v1`，进一步拆分 CPU 图片预处理，主要耗时如下：

| 阶段 | mean(ms) | 说明 |
|---|---:|---|
| open + JPEG decode + RGB convert | 10.794 | PIL lazy load 后在访问像素时完成解码 |
| grid resize | 22.351 | 原图 resize 到动态 tile 网格大小 |
| thumbnail resize | 17.231 | 额外生成 `448x448` thumbnail |
| crop | 0.352 | 从 grid resize 结果裁 tile |
| ToTensor + Normalize + stack | 3.565 | 转 tensor 与 ImageNet normalize |
| total | 54.315 | micro-profile 单独测得，和完整 latency 脚本存在样本/系统噪声差异 |

`pil_draft_parallel_no_resize` 的思路是使用 `PIL.Image.draft()` 给 JPEG decoder 一个低分辨率解码提示。普通路径近似为：

```text
1920x1080 JPEG -> 全分辨率 RGB -> resize 到 1344x448 / 448x448
```

draft 路径近似为：

```text
1920x1080 JPEG -> decode 时降采样到约 960x540 RGB -> resize 到 1344x448 / 448x448
```

它快的原因是 JPEG 解码阶段少处理像素，后续 resize 的输入也更小。但它不是无损工程优化，因为 resize 的源图已经变了，高频细节和插值结果都会改变。

Latency 结果：

| backend | 图片预处理 CPU(ms) | e2e GPU(ms) | 完整 latency(ms) | 与原 PIL tensor 等价 |
|---|---:|---:|---:|---|
| pil_parallel_no_resize + compact_v1 | 43.013 | 88.261 | 131.476 | 是 |
| pil_parallel_numpy + compact_v1 | 47.483 | 88.670 | 136.375 | 是 |
| pil_draft_parallel_no_resize + compact_v1 | 25.336 | 91.710 | 117.305 | 否 |

Tensor 差异：

| backend | max abs diff | mean abs diff |
|---|---:|---:|
| pil_parallel_no_resize | 0.000000 | 0.000000 |
| pil_parallel_numpy | 0.000000 | 0.000000 |
| pil_draft_parallel_no_resize | 0.736364 | 0.009364 |

结论：

- `pil_parallel_numpy` 虽然 tensor 等价，但实际比 `pil_parallel_no_resize` 慢，不推荐。
- 复用全局 resize 线程池的尝试也比每次创建局部线程池更慢，已回退。
- `pil_draft_parallel_no_resize` 将图片预处理从 `43.013 ms` 降到 `25.336 ms`，完整 latency 从 `131.476 ms` 降到 `117.305 ms`，是目前 CPU 侧收益最大的尝试。
- 但 draft 解码不是像素等价，输入 tensor 平均差异约 `0.00936`，必须跑 navtest PDMS 后才能判断是否可用；它应视为“速度换输入图像质量”的近似加速，而不是保精度优化。

真实车端部署注意：

- 自动驾驶在线链路里相机通常不会给 JPEG 文件，更常见的是 ISP 后的 `YUV/NV12/NV21/YUYV` 或某些链路中的 RGB/BGR buffer；Raw Bayer/RCCB 通常还要经过 ISP。
- JPEG/H.264/H.265/MJPEG 更多用于记录、回传或离线数据集存储，不一定是实时感知主链路输入。
- 因此 NAVSIM 上的 `JPEG decode + PIL resize` 主要是离线数据集复现成本，不完全等价于车端在线部署成本。
- 面向 910B / 昇腾部署时，更合理的方向是设计 `YUV/NV12 -> resize/crop/color convert/normalize -> NPU tensor` 的 DVPP/AIPP 前处理路径，而不是继续深挖 PIL/JPEG。

## FA2 下的 2B vs 8B

| 指标 | 2B-RL FA2 mean(ms) | 8B-RL FA2 mean(ms) | 8B / 2B |
|---|---:|---:|---:|
| VLM forward | 144.699 | 301.837 | 2.09x |
| diffusion planner | 88.864 | 89.810 | 1.01x |
| GPU 端到端 | 237.682 | 395.597 | 1.66x |
| 图片预处理 CPU | 92.291 | 75.323 | 0.82x |

核心结论：

- 本文档只保留 FA2 环境下的 latency 结果，因为 ReCogDrive 推理复现和后续优化均默认必须使用 FA2。
- FA2 下，2B 的 VLM 和 diffusion 延迟已经接近同一量级；如果继续优化 2B，只优化 VLM 的边际收益会下降，需要同时考虑 diffusion steps 或 action head 蒸馏。
- FA2 下，8B 的瓶颈仍然是 VLM forward，VLM 占 GPU 端到端的约 `76%`。
- CPU 图片预处理不受 FA2 影响，仍然有 `75-92 ms` 的额外开销；在线部署时需要异步预处理或预取，否则会显著拉高端到端 latency。

## 2B-RL + 视觉 Token Pruning

迁移方式：

- 没有直接接入 Prune2Drive 的 LLaVA 代码，而是在 ReCogDrive 的 InternVL wrapper 中实现可开关的视觉 token pruning。
- `agent.vlm_prune_keep_ratio=1.0` 时保持原始逻辑。
- `agent.vlm_prune_keep_ratio<1.0` 时，在 InternVL `extract_feature()` 得到每个动态 image patch 的 256 个视觉 token 后做 token selection，并同步减少 prompt 中 `<IMG_CONTEXT>` 数量。
- `tfps` 是 Prune2Drive 风格的 farthest-point token selection；`uniform` 是低开销均匀采样，用作速度上界参考。
- pruning 模式下 tokenizer 使用动态 padding，否则固定 `max_length=2800` 会抵消视觉 token 数减少带来的大部分收益。

50-sample latency 结果：

| 配置 | 视觉 token 数 | 输入 seq len | VLM mean(ms) | diffusion mean(ms) | GPU 端到端 mean(ms) | GPU 端到端下降 |
|---|---:|---:|---:|---:|---:|---:|
| 2B + FA2 baseline | 2304 | 约 2800 | 144.699 | 88.864 | 237.682 | - |
| 2B + FA2 + uniform 10% | 234 | 718.2 | 77.504 | 88.650 | 169.948 | 28.5% |
| 2B + FA2 + T-FPS 10% | 234 | 718.2 | 94.569 | 89.727 | 186.716 | 21.4% |
| 2B + FA2 + T-FPS 25% | 576 | 1060.2 | 124.623 | 89.012 | 216.092 | 9.1% |
| 2B + FA2 + T-FPS 50% | 1152 | 1636.2 | 182.384 | 89.885 | 276.448 | -16.3% |

相对 2B + FA2 baseline：

- `uniform 10%`：VLM 从 `144.7 ms` 降到 `77.5 ms`，加速 `1.87x`；GPU 端到端从 `237.7 ms` 降到 `169.9 ms`，加速 `1.40x`。
- `T-FPS 10%`：VLM 从 `144.7 ms` 降到 `94.6 ms`，加速 `1.53x`；GPU 端到端从 `237.7 ms` 降到 `186.7 ms`，加速 `1.27x`。
- `T-FPS 25%`：VLM 从 `144.7 ms` 降到 `124.6 ms`，加速 `1.16x`；GPU 端到端从 `237.7 ms` 降到 `216.1 ms`，加速 `1.10x`。
- `T-FPS 50%`：VLM 从 `144.7 ms` 上升到 `182.4 ms`，GPU 端到端从 `237.7 ms` 上升到 `276.4 ms`，没有 latency 收益。
- diffusion planner 基本不变，仍在 `~89 ms`；继续压低 2B latency 时，diffusion 会越来越成为下一个瓶颈。

解释：

- T-FPS 比 uniform 慢，主要是 token selection 本身需要计算 token 间相似度和 farthest-point 选择；它可能更保留视觉多样性，但需要通过 navtest PDMS 验证是否比 uniform 更稳。
- `T-FPS@0.50` 虽然减少了语言模型输入视觉 token，但当前实现要先跑完整 `extract_feature()`，再做 T-FPS 选择，并走手写 InternVL forward；50% 保留率下 token selection 和手写路径开销已经抵消了 token 数下降带来的收益。
- `T-FPS@0.25` 有小幅 latency 收益，但收益有限，因为 diffusion planner 仍然稳定占 `~89 ms`。
- 继续优化 T-FPS 应优先降低 selection 开销，或者改成低成本 token selection；否则较高 keep ratio 很难获得实际端到端加速。

### VLM 拆分 Profiling

为定位 `T-FPS@0.50` 反而变慢的原因，对 2B-RL 的 VLM forward 进一步拆分为 `vision encoder / token selection / language model`。该 profiling 会在原始 `vlm_cuda_ms` 之外额外跑一次手动拆分 forward，因此应看各子项的相对大小；`vlm_cuda_ms` 仍是原始完整 VLM forward 的端到端 CUDA event 计时。

| 配置 | visual tokens | seq len | VLM mean(ms) | vision encoder(ms) | token select(ms) | LLM(ms) | diffusion(ms) | e2e GPU(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B + FA2 baseline | 2304 | 2800.0 | 145.148 | 43.683 | 0.000 | 80.057 | 88.699 | 238.903 |
| 2B + FA2 + T-FPS 25% | 576 | 1060.2 | 125.208 | 43.170 | 41.337 | 31.926 | 88.798 | 217.965 |
| 2B + FA2 + T-FPS 50% | 1152 | 1636.2 | 178.636 | 43.547 | 78.411 | 44.255 | 89.164 | 272.188 |

拆分结论：

- Vision encoder 延迟基本不随 keep ratio 变化，三组都在 `~43.5 ms`，说明当前 pruning 没有减少视觉编码器计算。
- T-FPS@25% 将 LLM 从 `80.1 ms` 降到 `31.9 ms`，节省约 `48.1 ms`，但新增 token selection `41.3 ms`，所以 VLM 只净减少约 `20 ms`。
- T-FPS@50% 将 LLM 从 `80.1 ms` 降到 `44.3 ms`，节省约 `35.8 ms`，但 token selection 增加到 `78.4 ms`，净结果是 VLM 比 baseline 慢约 `33.5 ms`。
- 因此 `0.50` keep ratio 变慢的直接原因是 T-FPS selection 成本过高；它的计算开销超过了减少 LLM token 后得到的收益。
- 如果继续沿这个方向优化，优先考虑低成本 selection，例如 uniform、score/top-k、分块近似 FPS，或者在 vision encoder 内部提前 prune，而不是 `extract_feature()` 之后做全量 T-FPS。

### VLM 内部算子 Profiling

为判断 VLM 后续应该优先优化 `linear/GEMM`、attention 还是其他小算子，对当前主力配置做了 VLM 内部算子级 profiling。

测试配置：

- 模型：`ReCogDrive-VLM-2B / InternVL`
- 配置：`uniform pruning 0.50 + image max_num 6 / 3 tiles`
- 输入：`visual tokens=384`，`input seq len=870`
- 精度：VLM 为 `bfloat16`
- GPU：A800 GPU4
- 结果文件：
- 原始 profiler：`/data/zxz/HUAWEI/VLA/navsim_data/exp/profile/vlm_internal_2b_uniform050_maxnum6_gpu4.json`
- 高层 op 过滤版：`/data/zxz/HUAWEI/VLA/navsim_data/exp/profile/vlm_internal_2b_uniform050_maxnum6_gpu4_ops_only.json`

注意统计口径：

- CUDA event 是实际 forward 的端到端 GPU stream 时间，最适合和 latency 总表对齐。
- profiler 表中只统计高层 op 的 `self CUDA time`，用于判断算子占比；它不包含 PyTorch eager dispatch、kernel launch 间隙、stream idle、metadata/view 等无 CUDA kernel 的调度开销。
- 因此 profiler 各项相加不一定等于 CUDA event。LLM 中这个差值约 `6.35 ms`，说明除 GEMM 外还有明显的 launch/调度碎片。

Vision encoder 结果：

| 类别 | self CUDA time(ms) | 占高层 op self time |
|---|---:|---:|
| linear / GEMM | 9.015 | 52.3% |
| bicubic upsample / other | 2.367 | 13.7% |
| FlashAttention2 | 2.298 | 13.3% |
| pointwise | 1.224 | 7.1% |
| GELU / activation | 0.972 | 5.6% |
| LayerNorm | 0.793 | 4.6% |
| patch embedding conv | 0.455 | 2.6% |
| memory / shape / index | 0.115 | 0.7% |

Vision encoder CUDA event：`17.49 ms`。主要 top ops：

| op | self CUDA time(ms) | calls/iter |
|---|---:|---:|
| `aten::addmm` | 9.015 | 98 |
| `aten::upsample_bicubic2d` | 2.367 | 1 |
| `flash_attn::_flash_attn_varlen_forward` | 2.298 | 24 |
| `aten::gelu` | 0.972 | 25 |
| `aten::native_layer_norm` | 0.793 | 49 |

Vision encoder 结论：

- 主要瓶颈是 `linear/GEMM`，不是 attention；FA2 attention 只占高层 op self time 的约 `13.3%`。
- `aten::upsample_bicubic2d` 单次约 `2.37 ms`，占比接近 attention，值得单独排查。它大概率来自 ViT 位置编码/输入相关 interpolation；如果 tile shape 固定，后续可尝试缓存固定位置编码或避免每次动态 bicubic resize。
- patch embedding conv 只有 `0.46 ms`，不是主要优化点。

LLM 结果：

| 类别 | self CUDA time(ms) | 占高层 op self time |
|---|---:|---:|
| linear / GEMM | 15.497 | 65.6% |
| pointwise / RMSNorm 相关 | 3.922 | 16.6% |
| memory / copy / cat | 1.582 | 6.7% |
| FlashAttention2 | 1.275 | 5.4% |
| other | 0.729 | 3.1% |
| SiLU / activation | 0.624 | 2.6% |

LLM CUDA event：`29.98 ms`。高层 op self time 合计：`23.63 ms`，差值约 `6.35 ms`。主要 top ops：

| op | self CUDA time(ms) | calls/iter |
|---|---:|---:|
| `aten::mm` | 14.139 | 113 |
| `aten::mul` | 2.397 | 256 |
| `aten::addmm` | 1.356 | 84 |
| `flash_attn::_flash_attn_forward` | 1.275 | 28 |
| `aten::copy_` | 0.971 | 117 |
| `aten::add` | 0.789 | 170 |
| `aten::silu` | 0.624 | 28 |
| `aten::cat` | 0.611 | 57 |

LLM 结论：

- LLM 主要瓶颈是 `linear/GEMM`，约占高层 op self time 的 `65.6%`。
- attention 在 FA2 下只有 `~1.28 ms`，约 `5.4%`，继续切 attention backend 的收益很小。
- `mul/add/copy/cat/silu/RMSNorm` 等小算子和 launch 间隙累计明显。CUDA event 与高层 op self time 的 `~6.35 ms` 差值说明 PyTorch eager 的 kernel launch / 调度碎片约占 LLM forward 的 `20%`。
- 后续 LLM 优化优先级：减少 token/seq len > GEMM 量化或高效线性算子 > RMSNorm/SwiGLU/pointwise 融合。attention 不是当前优先方向。

整体判断：

- VLM 的两个主要模块都不是 attention 主导，而是 GEMM 主导。
- Vision encoder 的额外可疑点是 `bicubic upsample`，这部分比单个小算子更值得优先排查。
- 如果最终面向 910B，VLM 优化应重点关注静态 shape、GEMM 量化/高效 matmul、RMSNorm/SwiGLU/pointwise 融合，以及 position interpolation 这类可缓存常量路径。

## 2B-RL + Diffusion Steps Navtest 精度

测试设置：

- 数据集：NAVSIM navtest，完整 `12146` 个 scenario。
- 模型：2B-RL，FlashAttention2，单前视图，无视觉 token pruning。
- Diffusion：`sampling_method=ddim`，只改变 `agent.diffusion_num_inference_steps`。
- 运行结果：三组均 `12146/12146` 成功，`0` failed。

结果文件：

- DDIM 5 steps：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_rl_zxz/2026.05.17.09.51.16/2026.05.17.12.20.57.csv`
- DDIM 3 steps：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_rl_ddim3_zxz/2026.05.20.08.38.19/2026.05.20.10.19.46.csv`
- DDIM 1 step：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_rl_ddim1_zxz/2026.05.20.12.17.52/2026.05.20.14.12.52.csv`

| 配置 | DDIM steps | PDMS | 相对 5 steps | NC | DAC | EP | TTC | C | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B baseline | 5 | 0.904283 | - | 0.981023 | 0.976288 | 0.858269 | 0.950601 | 1.000000 | 0.966615 |
| 2B original | 3 | 0.907049 | +0.002766 | 0.981599 | 0.977770 | 0.861740 | 0.951507 | 1.000000 | 0.967191 |
| 2B original | 1 | 0.013528 | -0.890755 | 0.434711 | 0.114112 | 0.017215 | 0.165981 | 0.000000 | 0.170426 |

结论：

- `DDIM 3 steps` 相比默认 `5 steps` 没有精度下降，PDMS 反而高 `0.0028`；这个差异很小，但说明 2B 原始模型可以优先考虑把 diffusion steps 从 5 降到 3。
- `DDIM 1 step` 直接崩溃，PDMS 只有 `0.0135`，主要问题是 `comfort=0`，同时 `ego_progress`、`DAC`、`TTC`、`DDC` 都大幅下降。
- 因此 diffusion 采样步数的可用加速点目前是 `3 steps`，不是 `1 step`。

## Diffusion Planner 内部 Profiling

测试配置：

- 模型：`2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + OpenCV image backend`
- 功能级 profiling：`warmup=5`，有效样本 `30`
- block-level profiling：`warmup=5`，有效样本 `10`
- 脚本：`scripts/evaluation/profile_recogdrive_diffusion_cuda_event.py`
- 功能级结果：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_diffusion_profile_uniform050_ddim3_maxnum6_opencv_30.json`
- block-level 结果：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_diffusion_block_profile_uniform050_ddim3_maxnum6_opencv_10.json`

功能级拆分：

| 部分 | mean latency(ms) | 说明 |
|---|---:|---|
| profiled diffusion total | 57.693 | 细粒度打点后的总时间，略高于原始 benchmark |
| step0 DiT forward | 16.748 | 第 1 次 denoise 的 16-layer DiT |
| step1 DiT forward | 16.269 | 第 2 次 denoise 的 16-layer DiT |
| step2 DiT forward | 16.162 | 第 3 次 denoise 的 16-layer DiT |
| 3 次 DiT forward 合计 | 49.180 | diffusion 的绝对主瓶颈 |
| DDIM update / alpha / sigma / noise 等杂项 | 2.303 | DDIM 公式、extract、randn、clamp、trajectory update |
| action encoder | 1.042 | 当前 action + timestep 编码 |
| action decoder | 0.531 | DiT 输出到 action noise |
| condition encoder | 0.484 | VLM feature、history、ego status 编码 |
| fusion projector | 0.215 | history / VLM mean / action feature 融合 |

注意：

- 原始可比 benchmark 中 `diffusion_cuda_ms=53.893 ms`；profiling 总时间 `57.693 ms` 更高，主要来自大量 CUDA event 打点和同步开销。
- 因此上表应主要用于判断占比，不应替代总表中的端到端 latency。

block-level profiling 结论：

| DiT block 内部部分 | 16 blocks 合计 mean(ms) |
|---|---:|
| attention | 9.259 |
| FFN | 1.978 |
| modulate / norm / residual / adaLN 等 pointwise | 6.804 |

block-level profiling 会显著增加同步开销，带打点时 `profiled diffusion total=87.225 ms`，所以不能直接和原始 latency 对比。但它说明 DiT 内部并不是单一 FFN/GEMM 瓶颈，attention 与大量小算子、norm、modulation、residual 都有明显占比。

优化含义：

- 不减少 DDIM step 数时，diffusion 主要优化对象是每次 DiT forward，而不是 DDIM update 公式；DDIM 杂项只有 `~2.3 ms`。
- `DDIM 3 -> 2` 是 training-free 下最直接减少计算量的方法，理论上可以少一次 `~16 ms` DiT forward，但需要验证 PDMS。
- 面向 910B 部署时，应优先把 diffusion planner 整理成固定 shape、无动态控制流的独立子图，便于后续 CANN/ATC/MindIE 编译和算子融合。
- W8A8 只量化 Linear 未必能显著降低 diffusion 延迟，因为当前瓶颈还包含 attention、RMSNorm/AdaLN/modulate/residual 等大量非 Linear 小算子。

### Pointwise Fusion 尝试

进一步测试了 DiT block 中 pointwise 写法的低风险等价改写。该实验不改变权重、DDIM 采样步数、随机噪声分布或模型结构，只改变部分逐元素表达式：

- baseline：原始写法，`x * (1 + scale) + shift`，residual 为 `hidden + gate * output`。
- `addcmul_residual`：只把 gated residual 改成 `torch.addcmul(hidden, output, gate)`。
- `addcmul_pointwise`：把 modulation 改成 `torch.addcmul(shift, x, 1 + scale)`，同时 residual 使用 `torch.addcmul`。

结果文件：

- baseline：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/fusion_ablation_baseline_50.json`
- addcmul residual：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/fusion_ablation_addcmul_residual_50.json`
- addcmul pointwise：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/fusion_ablation_addcmul_pointwise_50.json`

| 版本 | diffusion(ms) | e2e GPU(ms) | 完整 latency(ms) |
|---|---:|---:|---:|
| baseline | 54.793 | 113.481 | 139.180 |
| addcmul residual only | 54.721 | 113.846 | 139.923 |
| addcmul pointwise | 45.999 | 103.277 | 129.471 |

输出等价性检查：

- 固定同一输入、同一随机种子，对比 baseline 与 `addcmul_pointwise`。
- `max_abs_diff=1.9073486328125e-06`
- `mean_abs_diff=1.738468853318409e-07`
- 首个轨迹点输出完全一致到打印精度：`[3.138187885284424, 0.009815216064453125, 0.0024843215942382812]`

结论：

- 只改 residual 基本没有收益，说明 `hidden + gate * output` 不是主要瓶颈。
- `addcmul_pointwise` 将 diffusion 从 `54.793 ms` 降到 `45.999 ms`，节省约 `8.8 ms`；e2e GPU 从 `113.481 ms` 降到 `103.277 ms`，节省约 `10.2 ms`。
- 输出差异只有 `1e-6` 量级，属于浮点表达式重排误差；因此这里不再单独跑 navtest PDMS。
- 该结果说明 diffusion 中 `modulate / norm / residual / adaLN` 一类 pointwise 表达式确实值得面向 910B 做静态子图和算子融合。

### Attention / RoPE 优化尝试

进一步检查 DiT attention 后，当前最确定的 training-free 优化点是 RoPE 路径，而不是 SDPA 本身：

- 原始 RoPE 每个 attention forward 都构造 `position_ids=torch.arange(N_q)`，再调用 `rotary_embedder()`，内部包含 `position_ids.max().item()` 和 `gather`。
- diffusion planner 的 action query 长度固定为连续位置，因此可以直接切 `cos_cached[:, :, :N_q, :]` 和 `sin_cached[:, :, :N_q, :]`，避免动态 tensor 创建、CPU sync 风险和 gather。
- 该优化已经迁移到实际 `navsim/agents/recogdrive/blocks/attention.py`，并保留 fallback：如果传入的 rotary embedder 不是当前 cache 实现，则回退原始 `position_ids + rotary_embedder()` 路径。
- `scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py` 增加了 `--sdpa-backend {auto,flash,math,efficient,cudnn}`，用于后续在空卡上单独测试 SDPA backend；该开关只包住 diffusion planner，不影响 VLM。

低层等价性检查：

- 固定同一个 `Attention` 权重、输入和 RoPE cache，对比原始 `position_ids + gather` 路径与直接 slice 路径。
- `max_abs_diff=0.0`
- `mean_abs_diff=0.0`

已完成的 latency ablation 结果如下。注意：这批 attention 实验运行时 CPU/full latency 有明显噪声，下面优先看 `diffusion_cuda_ms` 和 `e2e_gpu_cuda_ms`。

| 版本 | diffusion mean(ms) | diffusion trimmed mean(ms) | e2e GPU mean(ms) | e2e GPU trimmed mean(ms) |
|---|---:|---:|---:|---:|
| baseline | 54.793 | 54.561 | 113.481 | 112.929 |
| baseline + RoPE slice | 52.558 | 52.004 | 116.982 | 109.887 |
| addcmul pointwise | 45.999 | 45.530 | 103.277 | 103.063 |
| addcmul pointwise + RoPE slice | 43.468 | 42.341 | 102.414 | 99.833 |
| addcmul pointwise + RoPE slice + RoPE addcmul | 42.936 | 42.011 | 105.060 | 99.930 |

结论：

- RoPE slice 对 diffusion 有稳定收益：baseline 上约 `2.2-2.6 ms`，叠加 `addcmul_pointwise` 后约 `2.5-3.2 ms`。
- `RoPE addcmul` 相比普通 RoPE slice 的收益很小，且 e2e GPU mean 更噪，不建议优先作为主路径；默认实现保持普通表达式更稳。
- SDPA 本身目前仍使用 PyTorch 默认 `scaled_dot_product_attention` 自动 backend。由于 action query 长度只有 `8`，attention 的 GEMM/softmax 并不是唯一瓶颈，强制 backend 未必一定更快；需要在空卡上用新增 `--sdpa-backend` 逐项实测。
- 面向 910B，这类优化的价值不只是 A800 上节省几毫秒，更重要的是去掉动态 shape 辅助算子、`.item()` 和 gather，使 diffusion 子图更容易被 CANN/ATC/MindIE 做静态编译和融合。

SDPA backend ablation：

- 测试配置：`2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles + OpenCV + addcmul pointwise + RoPE slice`
- 测试 GPU：A800 GPU0，`warmup=5`，有效样本 `50`
- 结果文件：
- auto：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/sdpa_ablation_2b_uniform050_ddim3_maxnum6_opencv_addcmul_auto_50.json`
- math：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/sdpa_ablation_2b_uniform050_ddim3_maxnum6_opencv_addcmul_math_50.json`
- efficient：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/sdpa_ablation_2b_uniform050_ddim3_maxnum6_opencv_addcmul_efficient_50.json`
- flash / cudnn：运行失败，无有效 JSON。

| SDPA backend | 状态 | diffusion mean(ms) | diffusion trimmed mean(ms) | e2e GPU mean(ms) | e2e GPU trimmed mean(ms) |
|---|---|---:|---:|---:|---:|
| auto | 成功 | 41.812 | 41.721 | 98.274 | 98.012 |
| efficient | 成功 | 42.178 | 42.039 | 99.705 | 99.571 |
| math | 成功 | 48.734 | 48.509 | 105.795 | 105.610 |
| flash | 失败 | - | - | - | - |
| cudnn | 失败 | - | - | - | - |

补充说明：

- `auto` 是当前最优选择，diffusion 比强制 `efficient` 略快 `~0.3 ms`，e2e GPU 略快 `~1.4-1.6 ms`。
- `math` 明显更慢，不应作为部署路径。
- `flash` 和 `cudnn` 失败原因是 diffusion planner 当前 attention 的 Q/K/V 为 `float32`，PyTorch flash/cudnn SDPA 要求 Q/K/V 为 `float16` 或 `bfloat16`。
- 这说明单纯切 SDPA backend 没有明显额外空间；如果要继续挖 attention，需要考虑把 action head 的 attention 子路径安全降到 fp16/bf16 或针对 910B 做静态图融合，而不是强制 PyTorch backend。

### Diffusion Attention 内部拆分

为了确认 block-level profiling 中 `attention=9.259 ms` 的组成，进一步在 `Attention.forward()` 内部用 CUDA event 拆分：

- no fast-DDIM：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_diffusion_attention_internal_uniform050_ddim3_maxnum6_opencv_10.json`
- fast-DDIM：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_diffusion_attention_internal_fastddim_uniform050_ddim3_maxnum6_opencv_10.json`

测试配置：`2B + uniform@0.50 + DDIM3 + image max_num 6 / 3 tiles + OpenCV`。注意：attention 内部插入大量 CUDA event 会显著改变绝对耗时，因此下面主要看相对占比和调用数变化，不直接替代原始 latency。

no fast-DDIM 下，`attention` 是完整 attention module 调用，包含 `to_q/to_k/to_v`、`q_norm/k_norm`、RoPE、SDPA、`to_out` 和 reshape。按内部 component sum 的占比估算，映射到原 block-level `attention=9.259 ms` 后大致为：

| 部分 | 约等效耗时(ms) | 占比 |
|---|---:|---:|
| k_norm | 1.457 | 15.7% |
| SDPA | 1.442 | 15.6% |
| q_norm | 1.193 | 12.9% |
| RoPE q | 1.090 | 11.8% |
| to_q linear | 0.867 | 9.4% |
| to_k linear | 0.719 | 7.8% |
| to_v linear | 0.676 | 7.3% |
| to_out linear | 0.640 | 6.9% |
| RoPE cache/slice | 0.577 | 6.2% |
| RoPE k | 0.453 | 4.9% |
| reshape | 0.144 | 1.6% |

合并类别：

| 类别 | 占比 |
|---|---:|
| Q/K/V/out linear | 31.3% |
| Q/K norm | 28.6% |
| RoPE 相关 | 22.9% |
| SDPA 核心 attention | 15.6% |
| reshape | 1.6% |

开启 fast-DDIM 后，cross-attention 的 K/V 会提前通过 `build_cross_attention_kv_cache()` 缓存，DDIM 采样过程中不再重复计算 cross-attention 的 `to_k/to_v/k_norm`。调用数变化如下：

| 部分 | no fast-DDIM calls | fast-DDIM calls |
|---|---:|---:|
| to_q | 480 | 480 |
| to_k | 480 | 240 |
| to_v | 480 | 240 |
| q_norm | 480 | 480 |
| k_norm | 480 | 240 |
| SDPA | 480 | 480 |
| to_out | 480 | 480 |

fast-DDIM 下 attention 内部 component sum 占比变为：

| 部分 | 占比 |
|---|---:|
| SDPA | 19.0% |
| q_norm | 16.4% |
| RoPE q | 13.9% |
| to_q linear | 11.0% |
| k_norm | 9.2% |
| to_out linear | 7.9% |
| RoPE cache/slice | 7.3% |
| RoPE k | 5.7% |
| to_k linear | 4.0% |
| to_v linear | 3.8% |
| reshape | 1.9% |

fast-DDIM 的收益与代价：

- 收益：消除 cross-attention K/V projection 和 K norm 在每个 denoise step 内的重复计算；内部 profiling 中 attention component sum / DiT forward 从 `10.11 ms` 降到 `8.62 ms`，profiled total 从 `74.37 ms` 降到 `70.13 ms`。
- 精度代价：当前实现只缓存 DDIM steps 内不变的 `vl_features` 对应 K/V，等价性测试中输出差异为浮点重排量级，因此理论上不应带来独立 PDMS 风险。
- 显存代价：需要额外保存 cross-attention K/V cache，但当前 2B small diffusion 下这部分远小于 VLM 显存，不是主要瓶颈。
- 适用边界：只适合推理；如果后续模型让条件特征随 denoise step 变化，或者训练需要梯度，不应默认复用该 cache。
- 工程代价：多了一条 `forward_with_kv_cache` 路径，需要持续保证与原始 attention 路径数值一致。

结论：

- fast-DDIM 已经把 cross-attention K/V 重复计算这块优化掉，后续继续挖 attention 时，不应再把 K/V cache 当主要方向。
- 当前 attention 内部不是 SDPA 单点瓶颈；`q/k norm`、RoPE、Q/out linear 和 SDPA 都有明显占比。
- 面向 910B，attention 优化更合理的方向是固定 shape 子图、融合 q/k norm 与 RoPE、减少 pointwise/reshape/launch，而不是只替换 SDPA backend。

## 2B-RL + 视觉 Token Pruning Navtest 精度

测试设置：

- 数据集：NAVSIM navtest，完整 `12146` 个 scenario。
- 模型：2B-RL，FlashAttention2，单前视图。
- Pruning：uniform / T-FPS，每个动态 image patch 从 256 个视觉 token 中按比例保留。
- 运行结果：所有配置均 `12146/12146` 成功，`0` failed。

结果文件：

- uniform 10%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_prune_uniform_010_zxz/2026.05.18.01.39.14/2026.05.18.03.07.49.csv`
- T-FPS 10%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_prune_tfps_010_zxz/2026.05.18.01.37.42/2026.05.18.03.07.51.csv`
- T-FPS 25%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_prune_tfps_025_zxz/2026.05.18.03.14.53/2026.05.18.04.42.01.csv`
- T-FPS 50%：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_prune_tfps_050_zxz/2026.05.18.03.14.53/2026.05.18.04.53.43.csv`
- 2B baseline：`/data/zxz/HUAWEI/VLA/navsim_data/exp/recogdrive_agent_eval_2b_rl_zxz/2026.05.17.09.51.16/2026.05.17.12.20.57.csv`

| 配置 | 方法 | keep ratio | PDMS | 相对 baseline 下降 | NC | DAC | EP | TTC | C | DDC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B baseline | - | 1.00 | 0.904283 | - | 0.981023 | 0.976288 | 0.858269 | 0.950601 | 1.000000 | 0.966615 |
| Pruning | uniform | 0.10 | 0.671761 | 0.232522 | 0.858266 | 0.859625 | 0.660798 | 0.769142 | 1.000000 | 0.942244 |
| Pruning | T-FPS | 0.10 | 0.691985 | 0.212298 | 0.902437 | 0.830726 | 0.669353 | 0.825539 | 1.000000 | 0.937222 |
| Pruning | T-FPS | 0.25 | 0.801383 | 0.102900 | 0.951383 | 0.900049 | 0.770379 | 0.894451 | 1.000000 | 0.956652 |
| Pruning | T-FPS | 0.50 | 0.866982 | 0.037301 | 0.971637 | 0.947308 | 0.828356 | 0.930677 | 1.000000 | 0.963568 |

结论：

- `uniform@0.10` 和 `T-FPS@0.10` 精度损失都过大，不适合作为默认加速配置。
- 同样 10% token budget 下，`T-FPS@0.10` 的 PDMS 比 `uniform@0.10` 高 `0.0202`，说明 diversity selection 对精度有帮助，但当前 T-FPS 的 latency 开销较高。
- `T-FPS@0.25` 相比 10% 明显恢复精度，但 PDMS 仍下降约 `0.103`，主要损失来自 DAC、EP 和 TTC。
- `T-FPS@0.50` 是当前更合理的 Pareto 点，PDMS 下降约 `0.037`，关键安全指标也更接近 baseline。
- 结合 latency 后，`T-FPS@0.50` 不是有效加速点；它精度好但比 baseline 更慢。`T-FPS@0.25` 有约 `9.1%` GPU 端到端加速，但 PDMS 下降约 `0.103`，速度-精度交换偏差。

## 复现命令

2B-RL + FA2：

```bash
CUDA_VISIBLE_DEVICES=5 \
OPENSCENE_DATA_ROOT=/data/zxz/HUAWEI/VLA/navsim_data \
NUPLAN_MAPS_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/maps \
NAVSIM_EXP_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/exp \
/data/zxz/condaenv/curious_vla/navsim/bin/python \
scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py \
  --model-size 2b \
  --warmup 5 \
  --num-samples 50 \
  --output /data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_cuda_event_50_fa2.json
```

2B-RL + FA2 + T-FPS 10%：

```bash
CUDA_VISIBLE_DEVICES=6 \
OPENSCENE_DATA_ROOT=/data/zxz/HUAWEI/VLA/navsim_data \
NUPLAN_MAPS_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/maps \
NAVSIM_EXP_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/exp \
/data/zxz/condaenv/curious_vla/navsim/bin/python \
scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py \
  --model-size 2b \
  --warmup 5 \
  --num-samples 50 \
  --prune-keep-ratio 0.10 \
  --prune-method tfps \
  --output /data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_prune_tfps_0.10_50.json
```

8B-RL + FA2：

```bash
CUDA_VISIBLE_DEVICES=6 \
OPENSCENE_DATA_ROOT=/data/zxz/HUAWEI/VLA/navsim_data \
NUPLAN_MAPS_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/maps \
NAVSIM_EXP_ROOT=/data/zxz/HUAWEI/VLA/navsim_data/exp \
/data/zxz/condaenv/curious_vla/navsim/bin/python \
scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py \
  --model-size 8b \
  --warmup 5 \
  --num-samples 50 \
  --output /data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_8b_cuda_event_50_fa2.json
```
