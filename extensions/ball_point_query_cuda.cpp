#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> bpq_cuda_forward(at::Tensor pcs, at::Tensor centroids, 
        float radius, size_t max_samples);

std::vector<at::Tensor> bpq_forward(at::Tensor pcs, at::Tensor centroids, float radius, size_t max_samples) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(centroids);
    return bpq_cuda_forward(pcs, centroids, radius, max_samples);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bpq_forward, "CUDA forward routine for ball point query");
}
