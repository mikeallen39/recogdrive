# ReCogDrive PIL 图片预处理优化交接

当前目标：在不明显牺牲 navtest PDMS 的前提下，继续降低 `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles` 配置的端到端 latency。此前纯 OpenCV backend 将图片预处理从约 `55.319 ms` 降到 `23.235 ms`，但 PDMS 从 PIL backend 的 `0.850495` 降到 `0.837151`，额外下降 `0.013344`。因此下一步更适合优先优化 PIL 路径，尽量保持原始数值分布。

## 当前相关配置

- repo：`/mnt/42_store/zxz/HUAWEI/VLA/my_code/recogdrive`
- 环境：`/data/zxz/condaenv/curious_vla/navsim`
- 数据环境变量：`OPENSCENE_DATA_ROOT=/data/zxz/HUAWEI/VLA/navsim_data`
- 当前主要配置：`2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles`
- 原 PIL latency：完整 latency `167.491 ms`，图片预处理 CPU `55.319 ms`，PDMS `0.850495`
- OpenCV latency：完整 latency `133.029 ms`，图片预处理 CPU `23.235 ms`，PDMS `0.837151`
- OpenCV + addcmul pointwise + RoPE slice 的 GPU 侧优化结果已在 `my_docs/recogdrive_analysis.md` 中记录。

## 为什么 PIL backend 慢

当前 PIL 路径位于 `navsim/agents/recogdrive/utils/internvl_preprocess.py`：

- `Image.open(image_file).convert("RGB")`
- `dynamic_preprocess()` 中将原图 resize 到动态 tile grid，并 crop 出每个 tile。
- `use_thumbnail=True` 且多 tile 时，还会额外生成一张 `448x448` thumbnail。
- 每个 tile 再走 `build_transform()`：`RGB convert check -> Resize(448,448, BICUBIC) -> ToTensor -> Normalize`
- 最后 `torch.stack(pixel_values)`

慢点主要是：

- PIL resize 是 CPU 侧操作，BICUBIC/默认插值成本高，并行度有限。
- `dynamic_preprocess()` 已经把 tile crop 成 `448x448`，后续 `torchvision.transforms.Resize(448,448)` 很可能是冗余 resize。
- 每张 tile 都走 Python/PIL/torchvision 对象链路，`ToTensor + Normalize + stack` 也有 CPU 开销。
- `Image.open` 本身是 lazy 的，真正 JPEG decode 通常发生在 `convert("RGB")` 或 resize 时，所以 profiling 上 decode/convert/resize 会耦合。

## 建议优先实验

### 1. `pil_no_resize` backend

最高优先级。保留原始 PIL decode、`convert("RGB")`、`dynamic_preprocess()`、crop 和 thumbnail 逻辑，只去掉 tile 后 `transform` 里的 `Resize(448,448, BICUBIC)`。

原因：

- `dynamic_preprocess()` crop 出来的 tile 已经是 `448x448`。
- thumbnail 也是 `448x448`。
- 因此后续 `Resize(448,448)` 理论上是 no-op 或近似 no-op，但仍有调度和插值开销。
- 只保留 `ToTensor + Normalize`，精度风险最低，最可能做到“加速但不掉 PDMS”。

实现建议：

- 在 `internvl_preprocess.py` 中增加 `build_transform(input_size, do_resize=True)` 或新增 `build_tensor_transform()`。
- `backend == "pil_no_resize"` 时使用不含 Resize 的 transform。
- 注意 `load_image()` 的 backend choices 也要同步加到：
- `scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py`
- `navsim/agents/recogdrive/recogdrive_agent.py` / hydra yaml 如果 navtest 要用该 backend
- `scripts/evaluation/profile_recogdrive_diffusion_cuda_event.py` 如需要 profiling

### 2. `pil_parallel_no_resize`

在 `pil_no_resize` 基础上，使用已有的 `dynamic_preprocess_parallel()`，并保留 no-resize transform。

原因：

- 现有 `pil_parallel` 只并行 grid resize 和 thumbnail resize。
- 如果再去掉冗余 transform resize，可能进一步降低 CPU 图片预处理。
- 数值路径可能与 PIL 原始路径非常接近，但要注意并行 resize 调用顺序不应改变输出。

### 3. transform 缓存 / 减少对象创建

当前 `load_image()` 每次都会调用 `build_transform(input_size)` 创建 torchvision transform。可缓存：

- `build_transform(input_size, do_resize=True)`
- `build_transform(input_size, do_resize=False)`

预期收益小于去掉 resize，但基本无精度风险。

### 4. PIL-SIMD 或更快 JPEG 后端

暂时不建议作为第一步，因为会改变环境依赖，且可能影响全局 Pillow 行为。除非前 3 个方向收益不够，再考虑：

- `Pillow-SIMD`
- `torchvision.io.decode_image`
- `turbojpeg`

这些方案速度可能更好，但数值路径会更接近 OpenCV 风险，需要重新跑 PDMS。

## 建议实验顺序

1. 实现 `pil_no_resize`。
2. 在空闲 GPU 上用原 latency 脚本测：

```bash
OPENSCENE_DATA_ROOT=/data/zxz/HUAWEI/VLA/navsim_data \
CUDA_VISIBLE_DEVICES=<idle_gpu> \
PYTHONPATH=. \
/data/zxz/condaenv/curious_vla/navsim/bin/python scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py \
  --model-size 2b \
  --num-samples 50 \
  --warmup 5 \
  --prune-keep-ratio 0.50 \
  --prune-method uniform \
  --diffusion-steps 3 \
  --image-max-num 6 \
  --image-backend pil_no_resize \
  --dit-pointwise-variant addcmul_pointwise \
  --output /data/zxz/HUAWEI/VLA/navsim_data/exp/latency/recogdrive_2b_uniform050_ddim3_maxnum6_pil_no_resize_addcmul_50.json
```

3. 先做像素/feature 差异检查。建议抽 20-50 张图片，对比 `pil` 和 `pil_no_resize` 输出 tensor：

- `max_abs_diff`
- `mean_abs_diff`
- `num_patches` 是否一致
- shape 是否一致

4. 如果 `pil_no_resize` 输出与 `pil` 几乎一致，再跑 navtest PDMS。
5. 如果 `pil_no_resize` 仍然慢，再实现并测试 `pil_parallel_no_resize`。

## 判断标准

- 如果 `pil_no_resize` 的 PDMS 接近 `0.850495`，且 CPU 图片预处理明显低于 `55 ms`，就优先采用它，而不是纯 OpenCV。
- 如果 `pil_no_resize` 仍然明显掉点，需要检查 torchvision `Resize(448,448)` 是否并非真正 no-op，可能对像素做了重采样/rounding。
- 如果 `pil_no_resize` 精度保持但速度收益有限，再测 `pil_parallel_no_resize` 和 transform cache。
- 纯 OpenCV 当前速度很好，但 PDMS 已确认额外下降 `0.013344`，除非后续找到更接近 PIL 的 OpenCV resize/normalize 变体，否则不应作为默认精度优先路径。

## 2026-05-25 已完成更新

已实现并测试：

- `pil_no_resize`
- `pil_parallel_no_resize`
- transform cache

实现文件：

- `navsim/agents/recogdrive/utils/internvl_preprocess.py`
- `scripts/evaluation/benchmark_recogdrive_latency_cuda_event.py`

像素等价性：

- 抽 20 张 navtest 前视图，对比原 `pil`。
- `pil_no_resize`、`pil_parallel`、`pil_parallel_no_resize` 均为 `max_abs_diff=0.0`、`mean_abs_diff=0.0`。
- 说明这些 PIL 优化 backend 与原 PIL 输出 tensor 完全一致，理论上不需要额外跑 PDMS。

Latency 结果：

| backend | 图片预处理 CPU mean(ms) | e2e GPU mean(ms) | 完整 latency mean(ms) |
|---|---:|---:|---:|
| pil | 79.704 | 96.547 | 176.441 |
| pil_no_resize | 52.412 | 95.710 | 148.319 |
| pil_parallel | 41.220 | 96.299 | 137.708 |
| pil_parallel_no_resize | 41.395 | 94.597 | 136.188 |

结果文件：

- `pil`：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_50.json`
- `pil_no_resize`：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_no_resize_50.json`
- `pil_parallel`：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_parallel_50.json`
- `pil_parallel_no_resize`：`/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_parallel_no_resize_50.json`

当前建议：

- 默认精度优先路径应使用 `pil_parallel_no_resize`，因为它与原 PIL tensor 完全等价，同时完整 latency 接近 OpenCV。
- 如果要跑 navtest，需要把 `pil_parallel_no_resize` 加到 agent 配置校验/命令参数路径中；当前 latency 脚本已支持该 backend。

## 注意事项

- latency 测试必须使用 CUDA event benchmark 脚本，且应在空闲 GPU 上跑。
- PDMS 可以在有任务的 GPU 上跑，但要保证不影响已有任务。
- 不要改默认 `pil` 行为，新增 backend 做 ablation，便于可复现对比。
- 目前 action head dtype 的 bf16/fp16 实验还未完全收尾；已观察到 `bf16` 出现 NaN，`fp16` 未提速且输出有可见差异，因此图片预处理优化优先级更高。
- 当前未提交改动包括 attention/RoPE、benchmark 参数、文档更新，以及 pycache/navsim.egg-info 未跟踪文件。提交时不要包含 `__pycache__` 和 `navsim.egg-info`。
