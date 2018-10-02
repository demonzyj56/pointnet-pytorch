/*
 * We use a simple implementation for farthest point sampling which involves to steps:
 * 1. Calculate the distance between any two points.
 * 2. Sort the distance for each point with descending order.
 */
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/copy.h>


template <typename scalar_t, typename index_t>
void vectorized_argsort_descending(const int n, const int m, scalar_t *d_data, index_t *d_idx) {
    thrust::device_ptr<scalar_t> data_ptr = thrust::device_pointer_cast<scalar_t>(d_data);
    thrust::device_ptr<index_t>  idx_ptr = thrust::device_pointer_cast<index_t>(d_idx);
    thrust::device_vector<int> d_segments(n*m);
    thrust::device_vector<scalar_t> d_data_copy(n*m);
    for (int i = 0; i < n; ++i) {
        thrust::fill_n(thrust::device, d_segments.begin()+m*i, m, i);
    }
    thrust::copy(data_ptr, data_ptr+n*m, d_data_copy.begin());
    thrust::stable_sort_by_key(thrust::device, d_data, d_data+n*m, d_segments.begin(),
            thrust::greater<scalar_t>());
    thrust::stable_sort_by_key(thrust::device, d_data_copy.begin(), d_data_copy.end(), idx_ptr,
            thrust::greater<scalar_t>());
    thrust::stable_sort_by_key(thrust::device, d_segments.begin(), d_segments.end(), idx_ptr);
}


template <typename scalar_t>
__global__ void dist_map_forward_kernel(const size_t batch_size, const size_t size_a, const size_t size_b,
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, float* __restrict__ dist_map) {
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
        dist_map[index] = static_cast<float>((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3));
    }
}


template <typename scalar_t>
__global__ void batch_sampling_kernel(const size_t batch_size, const size_t num_points, 
        const size_t num_centroids, const int* __restrict__ argmax_idx_ptr, bool* __restrict__ sampled,
        scalar_t* __restrict__ centroid_idx) {
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size; 
            index += blockDim.x * gridDim.x) {
        centroid_idx[index*num_centroids] = 0;
        for (int i = 1; i < num_centroids; ++i) {
            scalar_t cur_idx = centroid_idx[index*num_centroids+i-1];
            for (int j = 0; j < num_points; ++j) {
                int next_idx = argmax_idx_ptr[(index*num_points+cur_idx)*num_points+j];
                if (sampled[index*num_points+next_idx])
                    continue;
                centroid_idx[index*num_centroids+i] = next_idx;
                sampled[index*num_points+next_idx] = true;
                break;
            }
        }
        
    }
}

void fps_cuda_forward(at::Tensor pcs, at::Tensor out) {
    // create distance map
    const auto batch_size = pcs.size(0);
    const auto num_points = pcs.size(2);
    const auto num_centroids = out.size(1);
    thrust::device_vector<float> dist_map(batch_size*num_points*num_points);
    float *dist_map_ptr = thrust::raw_pointer_cast(dist_map.data());
    const int threads = 1024;
    const int blocks = (batch_size * num_points * num_points + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "dist_map_forward_kernel", ([&] {
        dist_map_forward_kernel<<<blocks, threads>>>(
            batch_size, num_points, num_points, pcs.data<scalar_t>(), pcs.data<scalar_t>(),
            dist_map_ptr);
    }));
    cudaDeviceSynchronize();
    thrust::device_vector<int> argmax_idx(batch_size*num_points*num_points);
    for (int i = 0; i < batch_size*num_points; ++i) {
        thrust::sequence(argmax_idx.begin()+i*num_points, argmax_idx.begin()+(i+1)*num_points);
    }
    int *argmax_idx_ptr = thrust::raw_pointer_cast(argmax_idx.data());
    vectorized_argsort_descending(batch_size*num_points, num_points, dist_map_ptr, argmax_idx_ptr);
    cudaDeviceSynchronize();
    thrust::device_vector<bool> sampled(batch_size*num_points, false);
    bool *sampled_ptr = thrust::raw_pointer_cast(sampled.data());
    AT_DISPATCH_INTEGRAL_TYPES(out.type(), "batch_sampling_kernel", ([&] {
        batch_sampling_kernel<<<1, batch_size>>>(batch_size, num_points, num_centroids,
                argmax_idx_ptr, sampled_ptr, out.data<scalar_t>());
    }));
    cudaDeviceSynchronize(); 
}

template <typename scalar_t>
__global__ void fps_one_pass_kernel(const size_t batch_size, const size_t num_points,
        const size_t num_centroids, const scalar_t* __restrict__ pcs, bool* __restrict__ sampled,
        int64_t* __restrict__ centroid_idx) {
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size; 
            index += blockDim.x * gridDim.x) {
        centroid_idx[index*num_centroids] = 0;
        for (int i = 1; i < num_centroids; ++i) {
            const auto cur_idx = centroid_idx[index*num_centroids+i-1];
            const auto x1 = pcs[(index*3+0)*num_points+cur_idx];
            const auto x2 = pcs[(index*3+1)*num_points+cur_idx];
            const auto x3 = pcs[(index*3+2)*num_points+cur_idx];
            scalar_t max_dist = 0.;
            int64_t max_idx = -1;
            for (int j = 0; j < num_points; ++j) {
                // if j is already selected, move forward
                if (sampled[index*num_points+j])
                    continue;
                const auto y1 = pcs[(index*3+0)*num_points+j];
                const auto y2 = pcs[(index*3+1)*num_points+j];
                const auto y3 = pcs[(index*3+2)*num_points+j];
                const auto dist = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
                if (dist > max_dist) {
                    max_dist = dist;
                    max_idx = j;
                }
            }
            centroid_idx[index*num_centroids+i] = max_idx;
            sampled[index*num_points+max_idx] = true;
        }
    }
}

void fps_cuda_forward_one_pass(at::Tensor pcs, at::Tensor out) {
    const auto batch_size = pcs.size(0);
    const auto num_centroids = out.size(1);
    const auto num_points = pcs.size(2);
    thrust::device_vector<bool> sampled(batch_size*num_points, false);
    bool *sampled_ptr = thrust::raw_pointer_cast(sampled.data());
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps_one_pass_kernel", ([&] {
        fps_one_pass_kernel<<<1, batch_size>>>(batch_size, num_points, num_centroids,
                pcs.data<scalar_t>(), sampled_ptr, out.data<int64_t>());
    }));
}
