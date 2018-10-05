/*
 * We use a simple implementation for farthest point sampling which involves to steps:
 * 1. Calculate the distance between any two points.
 * 2. Sort the distance for each point with descending order.
 * 
 * Some part of code is adapted from:
 *   https://stackoverflow.com/questions/28150098/how-to-use-thrust-to-sort-the-rows-of-a-matrix
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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Timer of debug purpose
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

template <typename scalar_t, typename index_t>
void vectorized_argsort_descending(const int n, const int m, scalar_t *d_data, 
        index_t *d_idx, int *d_segments) {
    thrust::device_ptr<scalar_t> data_ptr = thrust::device_pointer_cast<scalar_t>(d_data);
    thrust::device_ptr<index_t> idx_ptr = thrust::device_pointer_cast<index_t>(d_idx);
    thrust::device_ptr<int> segments_ptr = thrust::device_pointer_cast<int>(d_segments);
    thrust::stable_sort_by_key(thrust::device, d_data, d_data+n*m,
            thrust::make_zip_iterator(thrust::make_tuple(segments_ptr, idx_ptr)),
            thrust::greater<scalar_t>());
    thrust::stable_sort_by_key(thrust::device, segments_ptr, segments_ptr+n*m, idx_ptr);
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

__global__ void index_assignment_kernel(const int n, const int m, int* __restrict__ idx, 
        int* __restrict__ segments) {
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < n*m; index += blockDim.x * gridDim.x) {
        idx[index] = index % m;
        segments[index] = index / m;
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

    /* unsigned long long tic = dtime_usec(0); */
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "dist_map_forward_kernel", ([&] {
        dist_map_forward_kernel<<<blocks, threads>>>(
            batch_size, num_points, num_points, pcs.data<scalar_t>(), pcs.data<scalar_t>(),
            dist_map_ptr);
    }));
    cudaDeviceSynchronize();
    /* std::cout << "dist map time: " << dtime_usec(tic)/(float)USECPSEC << "s" << std::endl; */

    /* tic = dtime_usec(0); */
    // Two auxiliary pointers for argsort.
    thrust::device_vector<int> argmax_idx(batch_size*num_points*num_points);
    int *argmax_idx_ptr = thrust::raw_pointer_cast(argmax_idx.data());
    thrust::device_vector<int> segments_aux(batch_size*num_points*num_points);
    int *segment_aux_ptr = thrust::raw_pointer_cast(segments_aux.data());
    index_assignment_kernel<<<blocks, threads>>>(batch_size*num_points, num_points, 
            argmax_idx_ptr, segment_aux_ptr);
    cudaDeviceSynchronize();
    vectorized_argsort_descending(batch_size*num_points, num_points, dist_map_ptr, 
            argmax_idx_ptr, segment_aux_ptr);
    cudaDeviceSynchronize();
    /* std::cout << "argsort time: " << dtime_usec(tic)/(float)USECPSEC << "s" << std::endl; */

    /* tic = dtime_usec(0); */
    thrust::device_vector<bool> sampled(batch_size*num_points, false);
    bool *sampled_ptr = thrust::raw_pointer_cast(sampled.data());
    AT_DISPATCH_INTEGRAL_TYPES(out.type(), "batch_sampling_kernel", ([&] {
        batch_sampling_kernel<<<1, batch_size>>>(batch_size, num_points, num_centroids,
                argmax_idx_ptr, sampled_ptr, out.data<scalar_t>());
    }));
    cudaDeviceSynchronize(); 
    /* std::cout << "batch sampling time: " << dtime_usec(tic)/(float)USECPSEC << "s" << std::endl; */
}

// A simpler implementation without arg-sorting the distance.
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



// We assume for now that num_points <= blockDim.x.
template <typename scalar_t>
__global__ void fps1_kernel(int batch_size, int num_points, int num_centroids, const scalar_t *pcs,
        int64_t *centroid_idx) {
    // shared memory
    extern __shared__ float sdata[];
    float *dists = sdata;
    int64_t *max_idx = (int64_t *)&dists[blockDim.x];
    bool *sampled = (bool *)&max_idx[blockDim.x];

    int64_t batch = blockIdx.x;
    int64_t tid = threadIdx.x;

    // loop over all centroids
    centroid_idx[batch*num_centroids] = 0;
    sampled[tid] = (tid == 0) ? true : false;
    __syncthreads();
    for (int64_t i = 1; i < num_centroids; ++i) {
        int64_t prev_idx = centroid_idx[batch*num_centroids+i-1];
        if (tid < num_points) {
            scalar_t x1 = pcs[(batch*3+0)*num_points+prev_idx];
            scalar_t x2 = pcs[(batch*3+1)*num_points+prev_idx];
            scalar_t x3 = pcs[(batch*3+2)*num_points+prev_idx];
            scalar_t y1 = pcs[(batch*3+0)*num_points+tid];
            scalar_t y2 = pcs[(batch*3+1)*num_points+tid];
            scalar_t y3 = pcs[(batch*3+2)*num_points+tid];
            dists[tid] = static_cast<float>((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3));
        } else {
            dists[tid] = 0.;
        }
        max_idx[tid] = tid;
        __syncthreads();

        // do reduction!
        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                // four cases depending whether each point is sampled or not.
                if (!sampled[max_idx[tid+s]]) {
                    if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+s])) {
                        dists[tid] = dists[tid+s];
                        max_idx[tid] = max_idx[tid+s];
                    }
                }
            }
            __syncthreads();
        }

        // write result to global memory
        if (tid == 0) {
            centroid_idx[batch*num_centroids+i] = max_idx[0];
        }

        // update sampled record
        if (tid == max_idx[0]) {
            sampled[max_idx[0]] = true;
        }
        __syncthreads();

    }
}

void fps_cuda_1(at::Tensor pcs, at::Tensor out) {
    int batch_size = pcs.size(0);
    int num_centroids = out.size(1);
    int num_points = pcs.size(2);
    int threads = 1024;
    int smem_size = (sizeof(float) + sizeof(int64_t) + sizeof(bool)) * threads;
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
        fps1_kernel<<<batch_size, threads, smem_size>>>(batch_size, num_points, num_centroids,
                pcs.data<scalar_t>(), out.data<int64_t>());
    }));
}

template <typename scalar_t, unsigned int blockSize>
__global__ void fps2_kernel(int batch_size, int num_points, int num_centroids, const scalar_t *pcs,
        int64_t *centroid_idx) {
    // shared memory
    __shared__ float dists[blockSize];
    __shared__ int64_t max_idx[blockSize];
    __shared__ bool sampled[blockSize];

    int64_t batch = blockIdx.x;
    int64_t tid = threadIdx.x;

    // loop over all centroids
    centroid_idx[batch*num_centroids] = 0;
    sampled[tid] = (tid == 0) ? true : false;
    __syncthreads();
    for (int64_t i = 1; i < num_centroids; ++i) {
        int64_t prev_idx = centroid_idx[batch*num_centroids+i-1];
        if (tid < num_points) {
            scalar_t x1 = pcs[(batch*3+0)*num_points+prev_idx];
            scalar_t x2 = pcs[(batch*3+1)*num_points+prev_idx];
            scalar_t x3 = pcs[(batch*3+2)*num_points+prev_idx];
            scalar_t y1 = pcs[(batch*3+0)*num_points+tid];
            scalar_t y2 = pcs[(batch*3+1)*num_points+tid];
            scalar_t y3 = pcs[(batch*3+2)*num_points+tid];
            dists[tid] = static_cast<float>((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3));
        } else {
            dists[tid] = 0.;
        }
        max_idx[tid] = tid;
        __syncthreads();

        // do reduction (fully unrolled!)
        if ((blockSize >= 1024) && (tid < 512)) {
            if (!sampled[max_idx[tid+512]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+512])) {
                    dists[tid] = dists[tid+512];
                    max_idx[tid] = max_idx[tid+512];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 512) && (tid < 256)) {
            if (!sampled[max_idx[tid+256]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+256])) {
                    dists[tid] = dists[tid+256];
                    max_idx[tid] = max_idx[tid+256];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 256) && (tid < 128)) {
            if (!sampled[max_idx[tid+128]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+128])) {
                    dists[tid] = dists[tid+128];
                    max_idx[tid] = max_idx[tid+128];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 128) && (tid < 64)) {
            if (!sampled[max_idx[tid+64]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+64])) {
                    dists[tid] = dists[tid+64];
                    max_idx[tid] = max_idx[tid+64];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 64) && (tid < 32)) {
            if (!sampled[max_idx[tid+32]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+32])) {
                    dists[tid] = dists[tid+32];
                    max_idx[tid] = max_idx[tid+32];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 32) && (tid < 16)) {
            if (!sampled[max_idx[tid+16]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+16])) {
                    dists[tid] = dists[tid+16];
                    max_idx[tid] = max_idx[tid+16];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 16) && (tid < 8)) {
            if (!sampled[max_idx[tid+8]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+8])) {
                    dists[tid] = dists[tid+8];
                    max_idx[tid] = max_idx[tid+8];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 8) && (tid < 4)) {
            if (!sampled[max_idx[tid+4]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+4])) {
                    dists[tid] = dists[tid+4];
                    max_idx[tid] = max_idx[tid+4];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 4) && (tid < 2)) {
            if (!sampled[max_idx[tid+2]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+2])) {
                    dists[tid] = dists[tid+2];
                    max_idx[tid] = max_idx[tid+2];
                }
            }
        }
        __syncthreads();

        if ((blockSize >= 2) && (tid < 1)) {
            if (!sampled[max_idx[tid+1]]) {
                if (sampled[max_idx[tid]] || (dists[tid] < dists[tid+1])) {
                    dists[tid] = dists[tid+1];
                    max_idx[tid] = max_idx[tid+1];
                }
            }
        }
        __syncthreads();

        // write result to global memory
        if (tid == 0) {
            centroid_idx[batch*num_centroids+i] = max_idx[0];
        }

        // update sampled record
        if (tid == max_idx[0]) {
            sampled[max_idx[0]] = true;
        }
        __syncthreads();

    }
}

int mylog2(int index) {
    int target = 0;
    while (index >>= 1)
        ++target;
    return target;
}

// assuming num_points <= 1024
void fps_cuda_2(at::Tensor pcs, at::Tensor out) {
    int batch_size = pcs.size(0);
    int num_centroids = out.size(1);
    int num_points = pcs.size(2);
    switch(mylog2(num_points-1)) {
        // since we assme num_points <= 1024.
        case 9:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 1024><<<batch_size, 1024>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 8:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 512><<<batch_size, 512>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 7:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 256><<<batch_size, 256>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 6:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 128><<<batch_size, 128>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 5:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 64><<<batch_size, 64>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 4:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 32><<<batch_size, 32>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 3:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 16><<<batch_size, 16>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 2:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 8><<<batch_size, 8>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 4><<<batch_size, 4>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
        case 0:
            AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
                fps2_kernel<scalar_t, 2><<<batch_size, 2>>>(batch_size, num_points, num_centroids,
                        pcs.data<scalar_t>(), out.data<int64_t>());
            }));
            break;
    }
}
