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
