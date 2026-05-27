#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> quantize_activation_per_token_int8_cuda(torch::Tensor input, double eps);
std::vector<torch::Tensor> rmsnorm_quantize_activation_per_token_int8_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double quant_eps);
std::vector<torch::Tensor> rmsnorm_static_quantize_activation_int8_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double static_scale);

std::vector<torch::Tensor> quantize_activation_per_token_int8(torch::Tensor input, double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  return quantize_activation_per_token_int8_cuda(input, eps);
}

std::vector<torch::Tensor> rmsnorm_quantize_activation_per_token_int8(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double quant_eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");
  TORCH_CHECK(input.size(1) == weight.size(0), "input hidden size must match RMSNorm weight size");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must have the same dtype");
  return rmsnorm_quantize_activation_per_token_int8_cuda(input, weight, rms_eps, quant_eps);
}

std::vector<torch::Tensor> rmsnorm_static_quantize_activation_int8(
    torch::Tensor input,
    torch::Tensor weight,
    double rms_eps,
    double static_scale) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");
  TORCH_CHECK(input.size(1) == weight.size(0), "input hidden size must match RMSNorm weight size");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must have the same dtype");
  TORCH_CHECK(static_scale > 0.0, "static_scale must be positive");
  return rmsnorm_static_quantize_activation_int8_cuda(input, weight, rms_eps, static_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "quantize_activation_per_token_int8",
      &quantize_activation_per_token_int8,
      "Fused per-token dynamic int8 activation quantization");
  m.def(
      "rmsnorm_quantize_activation_per_token_int8",
      &rmsnorm_quantize_activation_per_token_int8,
      "Fused RMSNorm and per-token dynamic int8 activation quantization");
  m.def(
      "rmsnorm_static_quantize_activation_int8",
      &rmsnorm_static_quantize_activation_int8,
      "Fused RMSNorm and static-scale int8 activation quantization");
}
