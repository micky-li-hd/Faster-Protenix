#include <torch/extension.h>
std::vector<at::Tensor> simt_forward(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int);
std::vector<at::Tensor> simt_backward(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&simt_forward);m.def("backward",&simt_backward);}
