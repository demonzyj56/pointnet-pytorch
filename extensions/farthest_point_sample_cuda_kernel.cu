#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// We assume for now that num_points <= blockDim.x.
// Fix some bugs.
template <typename scalar_t>
__global__ void fps_cuda_kernel(int batch_size, int num_points, int num_centroids, const scalar_t *pcs,
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
        } 
        max_idx[tid] = tid;
        __syncthreads();

        // do reduction!
        for (int64_t s = num_points / 2; s > 0; s >>= 1) {
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

void fps_cuda(at::Tensor pcs, at::Tensor out) {
    int batch_size = pcs.size(0);
    int num_centroids = out.size(1);
    int num_points = pcs.size(2);
    int threads = 1024;
    int smem_size = (sizeof(float) + sizeof(int64_t) + sizeof(bool)) * threads;
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "fps1_kernel", ([&] {
        fps_cuda_kernel<<<batch_size, threads, smem_size>>>(batch_size, num_points, num_centroids,
                pcs.data<scalar_t>(), out.data<int64_t>());
    }));
}
