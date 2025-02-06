// bias_query_kernel.cu

#include "bias_query_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) (((m) + (n)-1) / (n))

#ifndef EMPTY_KEY
#define EMPTY_KEY (-1)
#endif

namespace linear_ball_query_utils {

__forceinline__ __device__ int64_t xyz2key(
    int64_t x, int64_t y,
    int64_t z, int64_t b) {

    const int64_t bits_x = 18LL;
    const int64_t bits_y = 18LL;
    const int64_t bits_z = 18LL;
    const int64_t bits_b = 8LL;

    const int64_t shift_x = bits_y + bits_z + bits_b; // 18 + 18 + 8 = 44
    const int64_t shift_y = bits_z + bits_b;          // 18 + 8 = 26
    const int64_t shift_z = bits_b;                   // 8
    const int64_t shift_b = 0;

    const int64_t mask_x = (1LL << bits_x) - 1LL;
    const int64_t mask_y = (1LL << bits_y) - 1LL;
    const int64_t mask_z = (1LL << bits_z) - 1LL;
    const int64_t mask_b = (1LL << bits_b) - 1LL;

    return ((x & mask_x) << shift_x) |
           ((y & mask_y) << shift_y) |
           ((z & mask_z) << shift_z) |
           (b & mask_b);
}

} // namespace linear_ball_query_utils  <-- Add this to close the namespace

__device__ __host__ inline uint32_t splitmix64_hash(int64_t key) {
    uint64_t z = static_cast<uint64_t>(key) + 0x9e3779b97f4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return static_cast<uint32_t>(z);
}

__global__ void hash_insert_kernel(int n, const int64_t* __restrict__ d_keys,
                                   int64_t* __restrict__ d_hash_table_keys,
                                   int64_t* __restrict__ d_hash_table_values,
                                   int64_t HASH_SIZE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t key = d_keys[idx];
    int64_t value = idx;
    uint32_t hash = splitmix64_hash(key);
    uint32_t slot = hash & (HASH_SIZE - 1);

    while (true) {
        int64_t expected = EMPTY_KEY;
        int64_t desired = key;
        int64_t prev = atomicCAS((unsigned long long int*)&(d_hash_table_keys[slot]), expected, desired);
        if (prev == EMPTY_KEY || prev == key) {
            d_hash_table_values[slot] = value;
            break;
        }
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
}

void hash_insert_cuda_launcher(int n, const int64_t* d_keys,
                               int64_t* d_hash_table_keys,
                               int64_t* d_hash_table_values,
                               int64_t HASH_SIZE) {
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    hash_insert_kernel<<<blocks, threads>>>(n, d_keys, d_hash_table_keys, d_hash_table_values, HASH_SIZE);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("hash_insert_cuda_launcher launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void compute_counts_cuda_kernel(int m, int kernel_size,
                                           const int64_t* __restrict__ d_hash_table_keys,
                                           const int64_t* __restrict__ d_hash_table_values,
                                           int64_t HASH_SIZE,
                                           const int64_t* __restrict__ new_xyz,
                                           int64_t* __restrict__ counts) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    const int64_t* current_new_xyz = new_xyz + pt_idx * 4;

    int64_t grid_x = current_new_xyz[0];
    int64_t grid_y = current_new_xyz[1];
    int64_t grid_z = current_new_xyz[2];
    int64_t batch = current_new_xyz[3];

    int neighbor_count = 0;

    for (int dx = -kernel_size; dx <= kernel_size; ++dx) {
        for (int dy = -kernel_size; dy <= kernel_size; ++dy) {
            for (int dz = -kernel_size; dz <= kernel_size; ++dz) {
                int64_t key_temp = linear_ball_query_utils::xyz2key(
                    grid_x + dx, grid_y + dy, grid_z + dz, batch);

                uint32_t hash = splitmix64_hash(key_temp);
                uint32_t slot = hash & (HASH_SIZE - 1);

                while (true) {
                    int64_t current_key = d_hash_table_keys[slot];
                    if (current_key == key_temp) {
                        neighbor_count++;
                        break;
                    }
                    if (current_key == EMPTY_KEY) {
                        break;
                    }
                    slot = (slot + 1) & (HASH_SIZE - 1);
                }
            }
        }
    }
    counts[pt_idx] = neighbor_count;
}

void compute_counts_cuda_launcher(int m, int kernel_size,
                                  const int64_t* d_hash_table_keys,
                                  const int64_t* d_hash_table_values,
                                  int64_t HASH_SIZE,
                                  const int64_t* new_xyz,
                                  int64_t* counts) {
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    compute_counts_cuda_kernel<<<blocks, threads>>>(m, kernel_size, d_hash_table_keys, d_hash_table_values, HASH_SIZE,
                                                    new_xyz, counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_counts_cuda_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void fill_neighbor_indices_cuda_kernel(int m, int kernel_size,
                                                  const int64_t* __restrict__ d_hash_table_keys,
                                                  const int64_t* __restrict__ d_hash_table_values,
                                                  int64_t HASH_SIZE,
                                                  const int64_t* __restrict__ new_xyz,
                                                  int64_t* input_idx,
                                                  int64_t* output_idx,
                                                  const int64_t* counts,
                                                  const int64_t* offsets) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    const int64_t* current_new_xyz = new_xyz + pt_idx * 4;

    int64_t grid_x = current_new_xyz[0];
    int64_t grid_y = current_new_xyz[1];
    int64_t grid_z = current_new_xyz[2];
    int64_t batch = current_new_xyz[3];

    int64_t offset = offsets[pt_idx];
    int neighbor_idx = 0;

    for (int dx = -kernel_size; dx <= kernel_size; ++dx) {
        for (int dy = -kernel_size; dy <= kernel_size; ++dy) {
            for (int dz = -kernel_size; dz <= kernel_size; ++dz) {
                int64_t key_temp = linear_ball_query_utils::xyz2key(
                    grid_x + dx, grid_y + dy, grid_z + dz, batch);

                uint32_t hash = splitmix64_hash(key_temp);
                uint32_t slot = hash & (HASH_SIZE - 1);

                bool found = false;
                int64_t value = 0;

                while (true) {
                    int64_t current_key = d_hash_table_keys[slot];
                    if (current_key == key_temp) {
                        value = d_hash_table_values[slot];
                        found = true;
                        break;
                    }
                    if (current_key == EMPTY_KEY) {
                        break;
                    }
                    slot = (slot + 1) & (HASH_SIZE - 1);
                }

                if (found) {
                    input_idx[offset + neighbor_idx] = value;
                    output_idx[offset + neighbor_idx] = pt_idx;
                    neighbor_idx++;
                }
            }
        }
    }
}

void fill_neighbor_indices_cuda_launcher(int m, int kernel_size,
                                         const int64_t* d_hash_table_keys,
                                         const int64_t* d_hash_table_values,
                                         int64_t HASH_SIZE,
                                         const int64_t* new_xyz,
                                         int64_t* input_idx,
                                         int64_t* output_idx,
                                         const int64_t* counts,
                                         const int64_t* offsets) {
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    fill_neighbor_indices_cuda_kernel<<<blocks, threads>>>(m, kernel_size,
                                                           d_hash_table_keys, d_hash_table_values, HASH_SIZE,
                                                           new_xyz,
                                                           input_idx, output_idx,
                                                           counts, offsets);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fill_neighbor_indices_cuda_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
