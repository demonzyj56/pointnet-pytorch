#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> dist_map_cuda_forward(at::Tensor a, at::Tensor b);
std::vector<at::Tensor> dist_map_cuda_backward(at::Tensor grad_d, at::Tensor a, at::Tensor b);

std::vector<at::Tensor> dist_map_forward(at::Tensor a, at::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return dist_map_cuda_forward(a, b);
}

std::vector<at::Tensor> dist_map_backward(at::Tensor grad_d, at::Tensor a, at::Tensor b) {
    CHECK_INPUT(grad_d);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return dist_map_cuda_backward(grad_d, a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dist_map_forward, "CUDA Forward routine for dist map");
    m.def("backward", &dist_map_backward, "CUDA backward routine for dist map");
}
