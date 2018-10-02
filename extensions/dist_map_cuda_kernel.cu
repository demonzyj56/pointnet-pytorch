#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename scalar_t>
__global__ void dist_map_forward_kernel(const size_t batch_size, const size_t size_a, const size_t size_b,
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, scalar_t* __restrict__ dist_map) {
    const auto n = batch_size * size_a * size_b;
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const auto batch = index / (size_a * size_b);
        const auto h_index = index % (size_a * size_b);
        const auto h = h_index / size_b;
        const auto w = h_index % size_b;
        const auto x1 = a[(batch*3+0)*size_a+h];
        const auto x2 = a[(batch*3+1)*size_a+h];
        const auto x3 = a[(batch*3+2)*size_a+h];
        const auto y1 = b[(batch*3+0)*size_b+w];
        const auto y2 = b[(batch*3+1)*size_b+w];
        const auto y3 = b[(batch*3+2)*size_b+w];
        dist_map[index] = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
    }
}

std::vector<at::Tensor> dist_map_cuda_forward(at::Tensor a, at::Tensor b) {
    const auto batch_size = a.size(0);
    const auto size_a = a.size(2);
    const auto size_b = b.size(2);
    auto dist_map = at::zeros({batch_size, size_a, size_b}, a.options());

    const int threads = 1024;
    const int blocks = (batch_size * size_a * size_b + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(a.type(), "dist_map_cuda_forward", ([&] {
        dist_map_forward_kernel<scalar_t><<<blocks, threads>>>(
            batch_size, size_a, size_b, a.data<scalar_t>(), b.data<scalar_t>(),
            dist_map.data<scalar_t>()
        );
    }));
    return {dist_map};
}

template <typename scalar_t>
__global__ void dist_map_grad_a_kernel(const size_t batch_size, const size_t size_a, const size_t size_b,
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, const scalar_t* __restrict__ grad_d,
        scalar_t* __restrict__ grad_a) {
    const auto n = batch_size * size_a * 3;
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const auto batch = index / (size_a * 3);
        const auto h_index = index % (size_a * 3);
        const auto h = h_index / size_a;
        const auto w = h_index % size_a;
        for (auto j = 0; j < size_b; ++j) {
            const auto b_val = b[(batch*3+h)*size_b+j];
            const auto grad_d_val = grad_d[(batch*size_a+w)*size_b+j];
            grad_a[index] += 2. * (a[index] - b_val) * grad_d_val;
        }
    }
}

template <typename scalar_t>
__global__ void dist_map_grad_b_kernel(const size_t batch_size, const size_t size_a, const size_t size_b,
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, const scalar_t* __restrict__ grad_d,
        scalar_t* __restrict__ grad_b) {
    const auto n = batch_size * size_b * 3;
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const auto batch = index / (size_b * 3);
        const auto h_index = index % (size_b * 3);
        const auto h = h_index / size_b;
        const auto w = h_index % size_b;
        for (auto j = 0; j < size_a; ++j) {
            const auto a_val = a[(batch*3+h)*size_a+j];
            const auto grad_d_val = grad_d[(batch*size_a+j)*size_b+w];
            grad_b[index] += 2. * (b[index] - a_val) * grad_d_val;
        }
    }
}

std::vector<at::Tensor> dist_map_cuda_backward(at::Tensor grad_d, at::Tensor a, at::Tensor b) {
    const auto batch_size = a.size(0);
    const auto size_a = a.size(2);
    const auto size_b = b.size(2);
    auto grad_a = at::zeros_like(a);
    auto grad_b = at::zeros_like(b);
    const int threads = 1024;
    const int blocks_a = (batch_size * size_a * 3 + threads - 1) / threads;
    const int blocks_b = (batch_size * size_b * 3 + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(a.type(), "dist_map_cuda_backward", ([&] {
        dist_map_grad_a_kernel<scalar_t><<<blocks_a, threads>>>(
            batch_size, size_a, size_b, a.data<scalar_t>(), b.data<scalar_t>(), grad_d.data<scalar_t>(),
            grad_a.data<scalar_t>()
        );
    }));
    AT_DISPATCH_FLOATING_TYPES(b.type(), "dist_map_cuda_backward", ([&] {
        dist_map_grad_b_kernel<scalar_t><<<blocks_b, threads>>>(
            batch_size, size_a, size_b, a.data<scalar_t>(), b.data<scalar_t>(), grad_d.data<scalar_t>(),
            grad_b.data<scalar_t>()
        );
    }));
    return {grad_a, grad_b};
}
