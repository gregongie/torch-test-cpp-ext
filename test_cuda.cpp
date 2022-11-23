#include <torch/extension.h>

torch::Tensor circularFanbeamProjection_cuda(const torch::Tensor *image, torch::Tensor *sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

torch::Tensor circularFanbeamBackProjection_cuda(torch::Tensor *image, const torch::Tensor *sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

torch::Tensor circularFanbeamBackProjectionPixelDriven_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor circularFanbeamProjection(const torch::Tensor image, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(image);

  // allocate output sinogram tensor
  auto options = torch::TensorOptions().dtype(image.dtype()).device(image.device());
  auto sinogram = torch::zeros({image.size(0), nviews, nbins}, options);

  circularFanbeamProjection_cuda(image.data_ptr<float>(), sinogram.data_ptr<float>(), nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);

  return sinogram;
}

torch::Tensor circularFanbeamBackProjection(const torch::Tensor sinogram, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(sinogram);

  // allocate output sinogram tensor
  auto options = torch::TensorOptions().dtype(sinogram.dtype()).device(sinogram.device());
  auto image = torch::zeros({sinogram.size(0), nx, ny}, options);

  return circularFanbeamBackProjection_cuda(image.data_ptr<float>(), sinogram.data_ptr<float>(), nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

torch::Tensor circularFanbeamBackProjectionPixelDriven(const torch::Tensor sinogram, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(sinogram);
  return circularFanbeamBackProjectionPixelDriven_cuda(sinogram, nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circularFanbeamProjection", &circularFanbeamProjection, "Fanbeam Forward Projection");
  m.def("circularFanbeamBackProjection", &circularFanbeamBackProjection, "Fanbeam Back Projection");
  m.def("circularFanbeamBackProjectionPixelDriven", &circularFanbeamBackProjectionPixelDriven, "Fanbeam Back Projection, Pixel-driven");
}
