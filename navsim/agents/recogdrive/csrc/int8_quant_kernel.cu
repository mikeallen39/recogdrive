#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kQMax = 127;

__device__ __forceinline__ int8_t clamp_round_int8(float value) {
  value = nearbyintf(value);
  value = fminf(fmaxf(value, -static_cast<float>(kQMax)), static_cast<float>(kQMax));
  return static_cast<int8_t>(value);
}

template <typename scalar_t>
__global__ void quantize_activation_per_token_int8_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols,
    float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared_max[kThreads];
  const scalar_t* row_input = input + static_cast<int64_t>(row) * cols;
  int8_t* row_output = output + static_cast<int64_t>(row) * cols;

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]);
    local_max = fmaxf(local_max, fabsf(value));
  }

  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float scale = fmaxf(shared_max[0], eps) / static_cast<float>(kQMax);
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }

  float inv_scale = 1.0f / scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]) * inv_scale;
    row_output[col] = clamp_round_int8(value);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_quantize_activation_per_token_int8_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols,
    float rms_eps,
    float quant_eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreads];
  __shared__ float shared_max[kThreads];
  const scalar_t* row_input = input + static_cast<int64_t>(row) * cols;
  int8_t* row_output = output + static_cast<int64_t>(row) * cols;

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]);
    local_sum += value * value;
  }

  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(cols) + rms_eps);

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]) * inv_rms * static_cast<float>(weight[col]);
    local_max = fmaxf(local_max, fabsf(value));
  }

  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float scale = fmaxf(shared_max[0], quant_eps) / static_cast<float>(kQMax);
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }

  float inv_scale = 1.0f / scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]) * inv_rms * static_cast<float>(weight[col]) * inv_scale;
    row_output[col] = clamp_round_int8(value);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_static_quantize_activation_int8_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols,
    float rms_eps,
    float static_scale) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreads];
  const scalar_t* row_input = input + static_cast<int64_t>(row) * cols;
  int8_t* row_output = output + static_cast<int64_t>(row) * cols;

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]);
    local_sum += value * value;
  }

  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(cols) + rms_eps);
  float inv_scale = 1.0f / static_scale;
  if (threadIdx.x == 0) {
    scales[row] = static_scale;
  }
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_input[col]) * inv_rms * static_cast<float>(weight[col]) * inv_scale;
    row_output[col] = clamp_round_int8(value);
  }
}

}  // namespace

std::vector<torch::Tensor> quantize_activation_per_token_int8_cuda(torch::Tensor input, double eps) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  auto output = torch::empty({rows, cols}, input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows}, input.options().dtype(torch::kFloat32));

  const dim3 grid(rows);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "quantize_activation_per_token_int8_cuda",
      [&] {
        quantize_activation_per_token_int8_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                static_cast<int>(rows),
                static_cast<int>(cols),
                static_cast<float>(eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

std::vector<torch::Tensor> rmsnorm_static_quantize_activation_int8_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double static_scale) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  auto output = torch::empty({rows, cols}, input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows}, input.options().dtype(torch::kFloat32));

  const dim3 grid(rows);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "rmsnorm_static_quantize_activation_int8_cuda",
      [&] {
        rmsnorm_static_quantize_activation_int8_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                static_cast<int>(rows),
                static_cast<int>(cols),
                static_cast<float>(rms_eps),
                static_cast<float>(static_scale));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

std::vector<torch::Tensor> rmsnorm_quantize_activation_per_token_int8_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double quant_eps) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  auto output = torch::empty({rows, cols}, input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows}, input.options().dtype(torch::kFloat32));

  const dim3 grid(rows);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      input.scalar_type(),
      "rmsnorm_quantize_activation_per_token_int8_cuda",
      [&] {
        rmsnorm_quantize_activation_per_token_int8_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                static_cast<int>(rows),
                static_cast<int>(cols),
                static_cast<float>(rms_eps),
                static_cast<float>(quant_eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}
