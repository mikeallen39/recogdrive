# sgl_kernel 源码编译记录

本文档记录在 `/data/zxz/condaenv/curious_vla/navsim` 环境中尝试安装和源码编译 `sgl_kernel` 的完整过程、遇到的问题、解决方法和最终验证结果。

## 目标

为了在 RecogDrive 的 W8A8 真量化实验中替换 `torch._int_mm`，尝试使用 SGLang / SGL Kernel 中的 `int8_scaled_mm`。目标是获得 fused scaled int8 GEMM kernel，避免 `torch._int_mm` 路径中手动反量化、scale 乘法和额外 kernel launch 带来的高延迟。

## 当前环境

- Conda 环境：`/data/zxz/condaenv/curious_vla/navsim`
- Python：`3.9.23`
- PyTorch：`2.5.1+cu121`
- PyTorch CUDA：`12.1`
- 编译使用 CUDA toolkit：`/usr/local/cuda-12.2`
- 目标 GPU：A800，compute capability `sm80`
- 最终安装包：`sgl-kernel==0.0.3`
- 最终源码版本：本地 SGLang `v0.4.2` 的 `sgl-kernel`

## 问题 1：PyPI wheel ABI 不兼容

最开始尝试直接安装 PyPI 上已有的 `sgl-kernel` / `sglang-kernel` wheel，包括多个版本，例如 `0.3.3`、`0.2.x` 等。安装可以完成，但 import 失败。

典型错误：

```text
undefined symbol: _ZN3c108ListType3get...
```

这个错误说明 wheel 编译时链接的 PyTorch C++ ABI 与当前环境中的 PyTorch ABI 不一致。由于 `sgl_kernel` 是 C++/CUDA extension，Python 包版本匹配并不够，必须和当前 PyTorch 版本、CUDA、C++ ABI 兼容。

解决方法：

- 放弃直接使用 PyPI wheel。
- 清理已安装的坏包，避免 import 到错误版本。

清理命令：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/pip uninstall -y sgl-kernel sglang-kernel
rm -rf /data/zxz/condaenv/curious_vla/navsim/lib/python3.9/site-packages/sgl_kernel
rm -rf /data/zxz/condaenv/curious_vla/navsim/lib/python3.9/site-packages/sgl_kernel-*.dist-info
rm -rf /data/zxz/condaenv/curious_vla/navsim/lib/python3.9/site-packages/~gl_kernel*
```

## 问题 2：最新版 sgl-kernel 与当前环境不匹配

随后尝试使用本地最新版源码：

```text
/mnt/42_store/zxz/aiinfra/sglang/sgl-kernel
```

最新版 `sglang-kernel` 的 README / package metadata 对环境要求更高，主要问题是：

- 需要 Python `>=3.10`，但当前环境是 Python `3.9`。
- 目标依赖更接近 PyTorch `2.11`，但当前环境是 PyTorch `2.5.1+cu121`。
- 源码编译内容较多，构建时间很长。

尝试过程中补齐过构建依赖：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/pip install scikit-build-core
/data/zxz/condaenv/curious_vla/navsim/bin/pip install cmake==3.31.10
```

还修正过 CUDA toolkit 选择：

```bash
export PATH=/data/zxz/condaenv/curious_vla/navsim/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.2
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
export CUDAToolkit_ROOT=/usr/local/cuda-12.2
```

最新版源码可以开始编译，并编译到约 `[66/88]`，但 20 分钟超时，没有观察到明确源码错误。考虑到版本要求和编译成本，最终没有继续使用最新版。

解决方法：

- 改用较早、代码结构更简单、包含 `int8_scaled_mm` 的 SGLang `v0.4.2`。

## 问题 3：选择合适的源码版本

本地 SGLang repo 中较早版本并不都包含 `sgl-kernel`。检查后选择 `v0.4.2`，原因是：

- 已包含 `sgl-kernel` 子目录。
- 已包含 `int8_scaled_mm`。
- Python / PyTorch 约束比最新版宽松。
- 编译目标更少，更适合当前环境快速验证。

源码工作区：

```text
/tmp/sglang_v042_full/sgl-kernel
```

初始化过必要 submodule：

```text
sgl-kernel/3rdparty/cutlass
sgl-kernel/3rdparty/flashinfer
sgl-kernel/3rdparty/turbomind
sgl-kernel/3rdparty/cccl
```

## 问题 4：默认编译目标过多，构建成本高

`sgl-kernel v0.4.2` 默认会编译多个架构和多个 kernel。当前实验只需要 A800 上运行，因此只需要 `sm80`。

解决方法：

- 临时 patch `/tmp/sglang_v042_full/sgl-kernel/setup.py`。
- 只保留 `sm80` gencode。
- 限制并行编译任务数，避免 CPU / 内存压力过大。
- 显式指定 package，避免 package discovery 误判。

关键调整：

```python
packages=["sgl_kernel", "sgl_kernel.ops"]
package_dir={"sgl_kernel": "src/sgl-kernel"}
max_jobs=int(os.getenv("MAX_JOBS", "2"))
```

编译环境变量：

```bash
export PATH=/data/zxz/condaenv/curious_vla/navsim/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.2
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
export CUDAToolkit_ROOT=/usr/local/cuda-12.2
export MAX_JOBS=2
export SGL_KERNEL_ENABLE_BF16=1
export SGL_KERNEL_ENABLE_FP8=0
```

最终安装命令：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/pip install \
  --no-build-isolation \
  --no-deps \
  --ignore-requires-python \
  --force-reinstall \
  -v /tmp/sglang_v042_full/sgl-kernel \
  2>&1 | tee /tmp/sgl_kernel_build_v042_log.txt
```

最终编译成功，生成并安装：

```text
sgl_kernel-0.0.3-cp39-abi3-linux_x86_64.whl
```

## 问题 5：import 成功后还需要确认 CUDA kernel 可运行

安装成功后先检查 Python import：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/python - <<'PY'
import sgl_kernel
print("import ok", sgl_kernel.__file__)
print("has int8_scaled_mm", hasattr(sgl_kernel, "int8_scaled_mm"))
PY
```

结果：

```text
import ok /data/zxz/condaenv/curious_vla/navsim/lib/python3.9/site-packages/sgl_kernel/__init__.py
has int8_scaled_mm True
```

但仅 import 成功不代表 CUDA kernel 可用，所以继续在 GPU 上做最小调用测试。

## 问题 6：int8_scaled_mm 对 weight layout 有特殊要求

第一次最小调用失败：

```text
RuntimeError: mat_a must be a column major tensor
```

实际查看源码后发现报错信息有 typo，检查条件是：

```cpp
TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
```

真实要求是：

- `mat_a`：row-major，形状 `[M, K]`，`stride(1) == 1`
- `mat_b`：column-major，形状 `[K, N]`，`stride(0) == 1`
- `K` 必须是 `16` 的倍数
- `N` 必须是 `8` 的倍数
- `scales_a` 是 `[M]` contiguous float32
- `scales_b` 是 `[N]` contiguous float32

对于 PyTorch Linear 的 weight，原始量化 weight 通常保存为：

```text
weight_q: [out_features, in_features] contiguous
```

调用 `sgl_kernel.int8_scaled_mm` 时应该传：

```python
mat_b = weight_q.t()
```

注意不能写成：

```python
weight_q.t().contiguous()
```

因为 `.contiguous()` 会把 `mat_b` 变回 row-major，从而不满足 `stride(0) == 1`。

## 最终 CUDA 验证

在 GPU7 上执行最小验证：

```bash
CUDA_VISIBLE_DEVICES=7 /data/zxz/condaenv/curious_vla/navsim/bin/python - <<'PY'
import torch
from sgl_kernel import int8_scaled_mm

M, K, N = 64, 256, 128
x = torch.randn(M, K, device="cuda", dtype=torch.float16)
w = torch.randn(N, K, device="cuda", dtype=torch.float16)

xs = (x.abs().amax(dim=1).float() / 127).clamp(min=1e-6).contiguous()
ws = (w.abs().amax(dim=1).float() / 127).clamp(min=1e-6).contiguous()

xq = torch.round(x.float() / xs[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
wq = torch.round(w.float() / ws[:, None]).clamp(-127, 127).to(torch.int8).contiguous()

mat_b = wq.t()
print("xq stride", xq.stride(), "mat_b stride", mat_b.stride())

y = int8_scaled_mm(xq, mat_b, xs, ws, torch.float16, None)
torch.cuda.synchronize()

ref = x @ w.t()
err = (y - ref).abs()
print("ok", y.shape, y.dtype, "max_abs", err.max().item(), "mean_abs", err.mean().item())
PY
```

输出：

```text
xq stride (256, 1) mat_b stride (1, 256)
ok torch.Size([64, 128]) torch.float16 max_abs 0.73828125 mean_abs 0.12396240234375
```

结论：

- `sgl_kernel` 已经在当前 `navsim` 环境中源码编译成功。
- `int8_scaled_mm` 可以在 A800 / `sm80` 上运行。
- 后续接入 RecogDrive 时，应优先使用 `sgl_kernel.int8_scaled_mm`，失败时再 fallback 到 `torch._int_mm`。
- 接入时最重要的是保持 weight layout：保存 `[out_features, in_features]` contiguous int8，调用时传 `weight_q.t()`，不要对 transpose 后的 tensor 再做 contiguous。

## 后续接入建议

当前 `torch._int_mm` W8A8 路径在 RecogDrive workload 上明显变慢，原因是 activation 动态量化、int8 matmul、scale 反量化和 bias/reshape 等操作没有被充分融合。`sgl_kernel.int8_scaled_mm` 至少可以融合 int8 GEMM 与 scale 处理，理论上比 `torch._int_mm` 更接近可用的真 W8A8 推理路径。

建议下一步：

- 修改 `W8A8Int8Linear`，优先调用 `sgl_kernel.int8_scaled_mm`。
- 确保 weight int8 buffer 的 layout 与 SGL kernel 要求一致。
- 使用之前相同的 latency 脚本，在空闲 GPU 上重新测端到端 latency。
- 重点比较 `vlm_cuda_ms`、`vision_encoder_cuda_ms`、`language_model_cuda_ms` 和 `e2e_gpu_cuda_ms`，不要用繁忙 GPU 上的 wall time 判断量化收益。
