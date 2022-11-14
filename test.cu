#include <torch/extension.h>

torch::Tensor test_cuda_forward(torch::Tensor x);

torch::Tensor test_cuda_backward(torch::Tensor x);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor test_forward(torch::Tensor x) {
  CHECK_INPUT(x);
  return test_cuda_forward(x);
}

torch::Tensor test_backward(torch::Tensor x) {
  CHECK_INPUT(x);
  return test_cuda_backward(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &test_forward, "Test forward");
  m.def("backward", &test_backward, "Test backward");
}
