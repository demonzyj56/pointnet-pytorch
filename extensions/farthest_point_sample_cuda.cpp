#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void fps_cuda_forward(at::Tensor pcs, at::Tensor out);
void fps_cuda_forward_one_pass(at::Tensor pcs, at::Tensor out);
void fps_cuda_1(at::Tensor pcs, at::Tensor out);
void fps_cuda_2(at::Tensor pcs, at::Tensor out);
void fps_cuda_correct(at::Tensor pcs, at::Tensor out);

void fps_forward(at::Tensor pcs, at::Tensor out) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(out);
    fps_cuda_forward(pcs, out);
}

void fps_forward_one_pass(at::Tensor pcs, at::Tensor out) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(out);
    fps_cuda_forward_one_pass(pcs, out);
}

void fps1(at::Tensor pcs, at::Tensor out) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(out);
    fps_cuda_1(pcs, out);
}

void fps2(at::Tensor pcs, at::Tensor out) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(out);
    fps_cuda_2(pcs, out);
}

void fps3(at::Tensor pcs, at::Tensor out) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(out);
    fps_cuda_correct(pcs, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fps_forward, "CUDA forward routine for farthest point sampling");
    m.def("forward2", &fps_forward_one_pass, 
            "CUDA forward routine for farthest point sampling (simple implementation)");
    m.def("forward3", &fps1, "CUDA forward routine with reduction kernel.");
    m.def("forward4", &fps2, "CUDA forward routine with fully unrolled reduction kernel.");
    m.def("forward5", &fps3, "CUDA forward routine with reduction kernel.");
}
