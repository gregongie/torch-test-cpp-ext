#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor x) {
  auto s = torch::sigmoid(x);
  return (1 - s) * s;
}

torch::Tensor test_forward(torch::Tensor x) {
  auto s = torch::sigmoid(input);
  return s;
}

torch::Tensor test_backward(torch::Tensor x) {
  auto s = d_sigmoid(input);
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &test_forward, "Test forward");
  m.def("backward", &test_backward, "Test backward");
}
