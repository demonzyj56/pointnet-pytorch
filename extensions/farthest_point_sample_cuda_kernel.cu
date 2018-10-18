#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This is valid for num_points <= blockDim.x (=1024).
template <typename scalar_t>
__global__ void fps_cuda_kernel(int batch_size, int num_points, int num_centroids, const scalar_t *pcs,
        int64_t *centroid_idx) {
    // shared memory
    extern __shared__ float sdata[];
    float *max_dists = sdata;
    float *dists = &max_dists[blockDim.x];
    int64_t *max_idx = (int64_t *)&dists[blockDim.x];
    bool *sampled = (bool *)&max_idx[blockDim.x];

    int64_t batch = blockIdx.x;
    int64_t tid = threadIdx.x;

    // loop over all centroids
    centroid_idx[batch*num_centroids] = 0;
    dists[tid] = 1e38;
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
            scalar_t d = static_cast<float>((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3));
            if (d < dists[tid]) {
                dists[tid] = d;
            }
        } else {
            dists[tid] = -1.;
        }
        __syncthreads();

        max_dists[tid] = dists[tid];
        max_idx[tid] = tid;
        __syncthreads();

        // do reduction!
        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid < num_points) {
                // four cases depending whether each point is sampled or not.
                if (!sampled[max_idx[tid+s]]) {
                    if (sampled[max_idx[tid]] || (max_dists[tid] < max_dists[tid+s])) {
                        max_dists[tid] = max_dists[tid+s];
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

// Use external global memory.
template <typename scalar_t>
__global__ void fps_cuda_kernel_strided(int batch_size, int num_points, int num_centroids, const scalar_t *pcs,
        int64_t *centroid_idx, float *dist_buf, int64_t *idx_buf) {
    // shared memory
    extern __shared__ float sdata[];
    float *dists = sdata; // record distance from current point to all other points
    float *max_dists = &dists[blockDim.x]; // for reduction use
    int64_t *max_idx = (int64_t *)&max_dists[blockDim.x]; // record the corresponding index
    bool *sampled = (bool *)&max_idx[blockDim.x]; // record whether each point is sampled or not

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.y + threadIdx.x;

    // refer only to current batch
    dist_buf += batch * gridDim.y;
    idx_buf += batch * gridDim.y;

    if (gid == 0) {
        centroid_idx[batch*num_centroids] = 0;
    }
    dists[tid] = 1e38;
    sampled[tid] = ((gid == 0) || (gid >= num_points)) ? true : false;
    __syncthreads();
    for (int i = 1; i < num_centroids; ++i) {
        int prev_idx = centroid_idx[batch*num_centroids+i-1];
        if (gid < num_points) {
            scalar_t x1 = pcs[(batch*3+0)*num_points+prev_idx];
            scalar_t x2 = pcs[(batch*3+1)*num_points+prev_idx];
            scalar_t x3 = pcs[(batch*3+2)*num_points+prev_idx];
            scalar_t y1 = pcs[(batch*3+0)*num_points+gid];
            scalar_t y2 = pcs[(batch*3+1)*num_points+gid];
            scalar_t y3 = pcs[(batch*3+2)*num_points+gid];
            scalar_t d = static_cast<float>((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3));
            if (d < dists[tid]) {
                dists[tid] = d;
            }
        } else {
            dists[tid] = -1.;
        }
        __syncthreads();
        max_dists[tid] = dists[tid];
        max_idx[tid] = tid;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && gid < num_points) {
                // four cases depending whether each point is sampled or not.
                if (!sampled[max_idx[tid+s]]) {
                    if (sampled[max_idx[tid]] || (max_dists[tid] < max_dists[tid+s])) {
                        max_dists[tid] = max_dists[tid+s];
                        max_idx[tid] = max_idx[tid+s];
                    }
                }
            }
            __syncthreads();
        }

        // initialize
        if (tid == 0) {
            dist_buf[blockIdx.y] = sampled[max_idx[0]] ? -1. : max_dists[0];
            idx_buf[blockIdx.y] = blockDim.x * blockIdx.y + max_idx[0];
        }
        __syncthreads();

        // reduce on global memory
        if (gid == 0) {
            for (int j = 0; j < gridDim.y; ++j) {
                if (dist_buf[0] < dist_buf[j]) {
                    dist_buf[0] = dist_buf[j];
                    idx_buf[0] = idx_buf[j];
                }
            }
            centroid_idx[batch*num_centroids+i] = idx_buf[0];
        }
        __syncthreads();

        if (gid == centroid_idx[batch*num_centroids+i]) {
            sampled[tid] = true;
        }
        __syncthreads();

    }
}

void fps_cuda(at::Tensor pcs, at::Tensor out) {
    int batch_size = pcs.size(0);
    int num_centroids = out.size(1);
    int num_points = pcs.size(2);
    int threads = 1024;
    int smem_size = (2 * sizeof(float) + sizeof(int64_t) + sizeof(bool)) * threads;
    if (num_points <= threads) {
        AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
            fps_cuda_kernel<<<batch_size, threads, smem_size>>>(batch_size, num_points, num_centroids,
                    pcs.data<scalar_t>(), out.data<int64_t>());
        }));
    } else {
        int stride = (num_points + threads - 1) / threads;
        dim3 blocks(batch_size, stride, 1);
        float *d_buf = NULL;
        int64_t *d_idx = NULL;
        cudaMalloc((void **)&d_buf, batch_size*stride*sizeof(float));
        cudaMalloc((void **)&d_idx, batch_size*stride*sizeof(int64_t));
        AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps2_kernel", ([&] {
            fps_cuda_kernel_strided<<<blocks, threads, smem_size>>>(batch_size, num_points, num_centroids,
                    pcs.data<scalar_t>(), out.data<int64_t>(), d_buf, d_idx);
        }));
        cudaFree(d_buf);
        cudaFree(d_idx);
    }
}

void fps_cuda_strided(at::Tensor pcs, at::Tensor out) {
    int batch_size = pcs.size(0);
    int num_centroids = out.size(1);
    int num_points = pcs.size(2);
    int threads = 1024;
    int smem_size = (2 * sizeof(float) + sizeof(int64_t) + sizeof(bool)) * threads;
    int stride = (num_points + threads - 1) / threads;
    dim3 blocks(batch_size, stride, 1);
    float *d_buf = NULL;
    int64_t *d_idx = NULL;
    cudaMalloc((void **)&d_buf, batch_size*stride*sizeof(float));
    cudaMalloc((void **)&d_idx, batch_size*stride*sizeof(int64_t));
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps2_kernel", ([&] {
        fps_cuda_kernel_strided<<<blocks, threads, smem_size>>>(batch_size, num_points, num_centroids,
                pcs.data<scalar_t>(), out.data<int64_t>(), d_buf, d_idx);
    }));
    cudaFree(d_buf);
    cudaFree(d_idx);
}
