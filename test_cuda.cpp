#include <torch/extension.h>

torch::Tensor circularFanbeamProjection_cuda(torch::Tensor image, float ximageside, float yimageside,
                              float radius, float source_to_detector,
                              int nviews, float slen, int nbins) ;

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor circularFanbeamProjection(torch::Tensor image, float ximageside, float yimageside,
                              float radius, float source_to_detector,
                              int nviews, float slen, int nbins) {
  CHECK_INPUT(image);
  return circularFanbeamProjection_cuda(image, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &circularFanbeamProjection, "Circular Fanbeam Projection");
  // m.def("backward", &test_backward, "Test backward");
}
