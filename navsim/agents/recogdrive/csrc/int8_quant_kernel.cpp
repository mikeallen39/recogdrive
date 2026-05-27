#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> quantize_activation_per_token_int8_cuda(torch::Tensor input, double eps);

std::vector<torch::Tensor> quantize_activation_per_token_int8(torch::Tensor input, double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  return quantize_activation_per_token_int8_cuda(input, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "quantize_activation_per_token_int8",
      &quantize_activation_per_token_int8,
      "Fused per-token dynamic int8 activation quantization");
}
