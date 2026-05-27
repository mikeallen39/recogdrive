# ToMe 适配到 ReCogDrive Vision Encoder 的初步分析

## 结论

ToMe 可以尝试适配到 ReCogDrive 的 InternVL vision encoder，适配难度中等偏上。核心 matching / merge 逻辑可以参考：

```text
/mnt/42_store/zxz/HUAWEI/VLA/original_code/ToMe/tome/merge.py
```

但原始 ToMe patch 是给 timm ViT 写的，不能直接套到 InternVL。ReCogDrive 这里更稳的做法是：

```text
InternVL vision encoder 内部 merge token -> 后续 ViT 层用更少 token 计算 -> encoder 末尾 unmerge 回原始网格
```

这样可以降低 vision encoder 中后段计算量，同时保持 `extract_feature()` 的下游接口不变。

## 关键适配难点

InternVL 的 `extract_feature()` 强依赖 vision encoder 输出仍然是完整的方形 patch 网格：

```python
vit_embeds = vit_embeds[:, 1:, :]
h = w = int(vit_embeds.shape[1] ** 0.5)
vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
vit_embeds = self.mlp1(vit_embeds)
```

ReCogDrive 2B 的 vision 配置大致是：

```text
image_size = 448
patch_size = 14
hidden_size = 1024
num_hidden_layers = 24
num_attention_heads = 16
intermediate_size = 4096
norm_type = layer_norm
downsample_ratio = 0.5
原始 patch tokens = 32 x 32 = 1024
pixel_shuffle 后 image tokens = 16 x 16 = 256
```

如果直接在 vision encoder 内部 ToMe 后输出非方形 token 数，`pixel_shuffle + mlp1` 会被破坏。因此第一版不建议让 ToMe 改变最终输出 token 数，而应该在 encoder 末尾 unmerge 回 `cls + 1024 patch tokens`。

## 建议实现方案

第一版建议只做 training-free 的内部 token merge，不改 LLM prompt/token 数：

1. 参考 ToMe 的 `bipartite_soft_matching()` 和 `merge_wavg()` 实现 token matching 与 weighted average merge。
2. 给 `InternVisionEncoder` 增加 ToMe 状态，包括每层 merge 数量、token size、unmerge 函数列表。
3. 在指定层之后执行 merge，例如 layer 6 或 layer 12 之后。
4. 后续 encoder layer 在压缩后的 token 序列上运行。
5. encoder 结束后按相反顺序执行 unmerge，恢复到 `cls + 1024 patch tokens`。
6. 保持 `extract_feature()`、`pixel_shuffle()`、`mlp1`、LLM prompt 中 `<IMG_CONTEXT>` 数量不变。

第一版建议先不启用 ToMe 原论文里的 proportional attention。原因是当前 InternVL vision attention 使用 FlashAttention，proportional attention 需要给 attention score 加 `size.log()` bias，FlashAttention 路径不好直接插入，强行改会增加复杂度并可能降低速度。

## 与现有优化的关系

ToMe 作用在 vision encoder 内部，主要减少 ViT 后续层的 attention / MLP 计算量。

现有 `uniform pruning@0.50` 是在 `extract_feature()` 之后减少输入 LLM 的视觉 token 数，不能减少 vision encoder 的计算量。

因此两者作用位置不同，可以组合：

```text
ToMe: 降低 vision encoder latency
uniform pruning: 降低 LLM latency
```

不过 ToMe 第一版建议先在 vision BF16 上测试，不要立刻和 `w8a8_int8_layernorm_qkv_fc1` vision quant mode 混用。原因是二者都 wrapper 了 vision encoder layer，直接组合会增加调试复杂度。

## 预期收益

当前最优配置下 vision encoder 约 `16-17 ms`。ToMe 的收益会取决于开始 merge 的层数、最终 keep ratio 和 matching 开销。

粗略预期：

```text
保守配置：可能降低 1-2 ms
较激进 50% token 配置：可能降到 11-14 ms 区间
```

实际收益不会完全按 token ratio 线性下降，因为：

- 前若干层仍然跑完整 token。
- ToMe matching / scatter_reduce / unmerge 本身有额外开销。
- 最终仍要 unmerge 回完整网格，下游 `pixel_shuffle + mlp1` 不变。
- vision encoder 中 patch embedding、LayerNorm、residual 等非 token 二次复杂度部分也不会同比下降。

## 主要风险

- 原版 ToMe patch 面向 timm ViT，InternVL 需要单独 wrapper `InternVisionEncoderLayer` 和 `InternAttention`。
- ToMe 基于 key similarity 做全局 token merge，不显式保持空间邻域。自动驾驶场景的小目标、车道线、交通灯可能受影响。
- 不启用 proportional attention 时，merged token 的 size 信息不会进入后续 attention，精度可能弱于完整 ToMe。
- 如果最终 unmerge 回原始网格，LLM 端 token 数不变，所以端到端收益只来自 vision encoder，而不是 LLM。
- 如果不 unmerge，虽然可能进一步减少 LLM token，但需要重写 `extract_feature()` 的网格逻辑和 prompt token 数逻辑，风险明显更高。

## 建议实验配置

第一轮建议只测 latency，再根据结果决定是否跑 PDMS：

| 配置 | 目的 |
|---|---|
| ToMe after layer 6, final keep 75% | 保守配置，检查 PDMS 是否基本不掉 |
| ToMe after layer 6, final keep 50% | 主实验，观察 latency / PDMS tradeoff |
| ToMe after layer 12, final keep 50% | 降低精度风险，测试中后层压缩收益 |

如果 latency 收益小于 `1 ms`，不建议继续投入。如果 vision encoder 能稳定降低 `2-4 ms` 且 PDMS 掉点较小，再考虑与 LLM uniform pruning、DDIM3、fast DDIM 等当前最优配置组合。

