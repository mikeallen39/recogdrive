# Diffusion Planner Pointwise / Norm / AdaLN 融合方案

目标：在不改变模型权重、DDIM step、输入输出语义的前提下，继续降低 ReCogDrive diffusion planner 的推理延迟。当前重点是 DiT block 内的 `norm / AdaLN modulation / modulate / gated residual` 等 pointwise 小算子融合。

## 当前背景

当前主要优化配置：

- `2B uniform pruning 0.50 + DDIM 3 + image max_num 6 / 3 tiles`
- 图片 backend 推荐：`pil_parallel_no_resize`
- diffusion 已尝试：`addcmul_pointwise`
- RoPE 已优化：连续 position 直接 slice `cos_cached/sin_cached`

已有 profiling 结论：

- 原始 diffusion planner 约 `54-55 ms`
- `addcmul_pointwise` 后约 `46 ms`
- 叠加 RoPE slice / SDPA auto 后约 `41-42 ms`
- block-level profiling 显示 attention 和 `modulate / norm / residual / adaLN` 都有明显占比，FFN 不是唯一瓶颈。

这说明 pointwise 路径确实是有效优化方向，但单纯在 PyTorch eager 中继续改表达式，收益会逐渐变小。后续应转向固定 shape 子图和算子融合。

## 主要瓶颈形态

DiT block 典型计算流程：

```text
mod_params = adaLN_modulation(conditioning)
shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = chunk(mod_params)

normed = norm1(hidden)
modulated = normed * (1 + scale_attn) + shift_attn
attn_out = attention(modulated, ...)
hidden = hidden + gate_attn * attn_out

normed = norm2(hidden)
modulated = normed * (1 + scale_ffn) + shift_ffn
ffn_out = ffn(modulated)
hidden = hidden + gate_ffn * ffn_out
```

当前问题：

- `B=1`、`action_horizon=8`、hidden dim 小，很多 pointwise kernel 很碎。
- 每个 DiT forward 有 `16` 个 blocks，每个 DDIM step 跑一次 DiT，DDIM3 共重复 `48` 次 block。
- `chunk / unsqueeze / add / mul / addcmul / norm / residual` 单次不大，但 kernel launch 和显存读写累计明显。
- timestep 变化导致 AdaLN modulation 不能全部跨 step 缓存，但 history/VLM/ego 条件中有部分可以预计算。

## 分阶段方案

### Phase 1：PyTorch 等价重写，继续清理动态图开销

目标：在当前 A800/PyTorch 环境中做低风险 ablation，确认哪些表达式还有收益。

可做项：

- 保留现有 `addcmul_pointwise`，将其从 benchmark monkey-patch 迁移到正式可配置实现。
- 用 `expand` 替代不必要的 `repeat`，尤其是：
- `vl_features.mean(1).unsqueeze(1).repeat(1, action_horizon, 1)`
- `history_embeds.unsqueeze(1).repeat(...)`
- 预创建固定 timestep/index tensor，减少 DDIM loop 内 `torch.full`。
- 检查 `chunk + unsqueeze` 是否能改成 view/slice，减少临时对象。
- 避免 diffusion forward 内的 `.item()`、动态 shape 分支、重复 dtype/device 查询。

预期收益：

- A800 上可能只有 `1-3 ms`，但能为后续静态图做准备。

验证：

- 固定同一输入和随机种子，对比原始 action head 输出。
- 记录 `max_abs_diff`、`mean_abs_diff` 和首个轨迹点。
- 如果误差只有 `1e-6` 量级，可视为浮点表达式重排，不单独跑 PDMS。

### Phase 2：拆出固定 shape diffusion 子图

目标：把 diffusion planner 推理路径整理成静态、可编译、无动态控制流的独立模块。

固定输入：

```text
vl_features: [1, seq_len, hidden_dim]
his_traj_features / history_trajectory: fixed shape
ego_status_features: fixed shape
initial_noise/current_actions: [1, 8, 3]
```

固定配置：

```text
B = 1
action_horizon = 8
hidden_dim = 384 for 2B
num_layers = 16
DDIM steps = 3
sampling_method = ddim
```

需要清理：

- Python `for i in range(self.ddim_steps)` 可保留为固定 unroll，或显式展开成 3 次 step。
- 固定 `ddim_t`、`index_batch` tensor，作为 buffer。
- 将 `make_timesteps()` 和 `extract()` 中动态 shape 逻辑改成固定 shape。
- 将不随 step 变化的条件编码提前到 loop 外：
- `his_traj_encoder`
- `ego_status_encoder`
- `vl_features.mean(1)`
- cross-attention K/V cache
- 保证所有 tensor dtype/device 在进入子图前确定。

预期收益：

- PyTorch eager 未必显著变快。
- 对 910B/CANN/ATC/MindIE 更重要，因为静态图后编译器才有机会做 pointwise fusion、constant folding、buffer reuse。

验证：

- 先在 A800 上做输出等价。
- 再用 CUDA event 测 diffusion-only latency 和 full latency。
- 输出差异接受标准：`max_abs_diff <= 1e-5` 优先；如果更大，需要跑小规模/完整 navtest PDMS。

### Phase 3：融合 norm + modulation + residual pattern

目标：把下面几类 pattern 变成编译器容易识别或手工融合的形式。

候选融合 pattern：

```text
norm(x) -> x * (1 + scale) + shift
hidden + gate * branch_out
norm(x) -> modulate -> attention/ffn
```

建议实现层级：

1. 先做 graph-friendly rewrite。
2. 再看 CANN/ATC/MindIE 是否自动融合。
3. 如果自动融合不足，再考虑自定义融合算子。

不建议优先做 CUDA/Triton kernel：

- A800 上可能有效，但最终目标是 910B。
- CUDA/Triton 不能直接迁移到 Ascend。
- 除非只是为了估计理论上限，否则工程优先级低。

910B 方向：

- 用固定 shape 子图导出或转换。
- 检查 CANN 图中是否出现大量 `Mul/Add/Unsqueeze/Cast/Reshape` 小算子。
- 优先让编译器融合 `Mul + Add + Addcmul` 类 pointwise。
- 对 RMSNorm/LayerNorm + modulation，如编译器不能融合，再考虑自定义 Ascend 算子或图 rewrite。

预期收益：

- PyTorch/A800：可能 `2-5 ms`。
- 910B 静态图：潜在收益更大，但依赖编译器融合效果。

### Phase 4：重新评估 attention 和 K/V cache

当前 RoPE slice 和 SDPA backend 已测试：

- `auto` 最优。
- `math` 明显慢。
- `flash/cudnn` 因 fp32 Q/K/V 不可用。
- action head 直接转 bf16 出现 NaN，fp16 未提速且输出差异可见。

下一步可做：

- 在固定 shape 子图中重新测试 cross-attention K/V cache。
- 只对 attention 内部做局部半精度需要谨慎，必须先做输出和 PDMS 验证。
- 不建议直接把整个 action head 转 bf16/fp16 作为默认方向。

## 推荐实施顺序

1. 将现有 `addcmul_pointwise` 正式化为 action head 可配置实现，而不是只在 latency script monkey-patch。
2. 做 Phase 1 的动态图清理：`repeat -> expand`、固定 timestep buffers、去掉重复 dtype/device 查询。
3. 新建 diffusion-only 等价性测试脚本，固定输入和随机种子，对比 baseline / optimized 输出。
4. 新建固定 shape `get_action_static_ddim3()`，先在 PyTorch eager 中等价验证。
5. 对 `get_action_static_ddim3()` 做 latency profiling，拆分每个 step 和每类 block。
6. 面向 910B，尝试导出/编译固定 shape 子图，观察编译后 graph fusion 情况。
7. 如果编译器没有融合 norm/modulate/residual，再考虑自定义融合算子或 graph rewrite。

## 测试指标

必须记录：

- diffusion CUDA event mean / median / trimmed mean
- e2e GPU CUDA event mean / median / trimmed mean
- full sync latency
- 输出等价性：`max_abs_diff`、`mean_abs_diff`
- 首个轨迹点输出

可选记录：

- 每个 DDIM step 的 DiT forward 时间
- block 内 attention / FFN / pointwise 时间
- CANN/ATC/MindIE 编译图中的算子数量和融合情况

## 风险

- 只做 PyTorch eager 表达式重写，收益可能很小。
- norm/modulate/residual 融合若改变浮点执行顺序，可能产生 `1e-6 ~ 1e-5` 误差，通常可接受，但需要记录。
- 半精度 action head 风险较高：当前 bf16 直接 NaN，fp16 未提速且输出差异明显。
- 910B 上的最终收益不能完全由 A800 PyTorch latency 预测，必须以 CANN/ATC/MindIE 编译结果为准。

## 当前判断

pointwise/norm/AdaLN 融合值得继续做，但正确路线不是继续堆 PyTorch 小技巧，而是：

```text
固定 shape diffusion 子图
-> 清理动态图和重复小算子
-> graph-friendly pointwise rewrite
-> 依赖 910B 编译器融合
-> 必要时自定义融合算子
```

如果只看短期 A800 实验，预期收益约 `2-5 ms`；如果面向 910B 静态部署，收益潜力更高，是 diffusion planner 后续优化的重点方向。

## 2026-05-26 实施记录

本轮已完成的代码改动：

- 将 `addcmul_pointwise` 从 latency benchmark 的 monkey-patch 迁移为 `LightningDiT.set_pointwise_variant()` 正式路径。
- `ReCogDriveAgent` 增加 `dit_pointwise_variant` 和 `fast_ddim_action` 配置，默认保持原行为；实际 navtest 可通过 Hydra override 打开。
- RoPE 中的 `(x * cos) + (rotate_half(x) * sin)` 改为 `torch.addcmul(...)` 表达式。
- `get_action_fast_ddim()` 中的部分 `repeat` 改为 `expand`，DDIM 小公式改为 `addcmul` 表达式，减少中间张量。

等价性验证：

- 小模型 DiT baseline vs `addcmul_pointwise`：`max_abs_diff=0.0`，`mean_abs_diff=0.0`。
- DDIM 公式重写：`max_abs_diff` 约 `9.54e-7`。
- 实际 2B agent 上 `get_action()` vs `get_action_fast_ddim()`，固定同一 `init_actions` 且 `deterministic=True`：`max_abs_diff=0.0`，`mean_abs_diff=0.0`。

Latency 结果，配置为 `2B uniform pruning 0.50 + DDIM3 + image max_num 6 / 3 tiles + pil_parallel_no_resize + addcmul_pointwise + sdpa auto`，A800 GPU7，50 samples：

| 配置 | diffusion mean | diffusion median | e2e GPU mean | 备注 |
| --- | ---: | ---: | ---: | --- |
| 旧记录，无 fast DDIM | `40.03 ms` | `39.98 ms` | `94.60 ms` | `/data/zxz/HUAWEI/VLA/navsim_data/exp/latency/pil_backend_2b_uniform050_ddim3_maxnum6_addcmul_pil_parallel_no_resize_50.json` |
| 正式 pointwise 路径，无 fast DDIM | `40.96 ms` | `40.60 ms` | `97.19 ms` | 与旧记录相比没有稳定收益，主要是表达式正式化而非新融合 |
| 正式 pointwise + fast DDIM | `38.70 ms` | `38.23 ms` | `95.94 ms` | diffusion 约快 `1.3 ms`，e2e 受 VLM/长尾波动影响不稳定 |

当前结论：

- 单纯 PyTorch eager 的 pointwise 表达式融合已经接近收益上限，继续堆 `addcmul` 只能拿到 `1-2 ms` 级别收益。
- `fast_ddim_action=true` 是目前最明确的 diffusion 侧 training-free 加速开关，主要收益来自 cross-attention K/V cache 和 loop-invariant 清理。
- `EtaFixed.__call__()` 的 `eta.item()` 路径尝试改成 GPU tensor expand 后，CUDA event 反而更慢；原因是原 CPU 同步不完全计入 CUDA event，而新写法增加了 GPU pointwise kernel launch，因此已回退。
- 如果目标是继续减少 kernel launch，下一步不应继续只改局部表达式，而应把 DDIM3 推理显式整理成固定 shape 子图，给 910B/CANN 编译器做跨算子 fusion 的机会。
