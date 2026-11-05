#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor x, torch::Tensor dy,
    torch::Tensor mean, torch::Tensor rstd, bool want_weight, bool want_bias, int64_t columns);
namespace width16 {
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,bool,int64_t);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine16_backward", &width16::affine_backward_cuda, "Ordered width16 affine gradients");
  m.def("affine_backward", &affine_backward_cuda, "Ordered direct LayerNorm affine gradients");
}
