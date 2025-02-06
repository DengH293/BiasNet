// bias_max_kernel_shared_memory.cu

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>    // For FLT_MAX
#include <cassert>
#include <algorithm>

__device__ void atomicMaxFloatIndex(float* address_f, int* address_i, float val, int idx) {
    unsigned int* address_as_ui = (unsigned int*)address_f;
    unsigned int old_ui = *address_as_ui, assumed_ui;

    do {
        assumed_ui = old_ui;
        float assumed_val = __uint_as_float(assumed_ui);
        if (!(assumed_val >= val)) {
            old_ui = atomicCAS(address_as_ui, assumed_ui, __float_as_uint(val));
            if (old_ui == assumed_ui) {
                atomicExch(address_i, idx);
                break;
            }
        } else {
            break;
        }
    } while (assumed_ui != old_ui);
}

namespace {

template <typename T, int K, int V, int C_in>
__global__ void BiasMaxForwardKernelA(
    const T* __restrict__ f,
    const T* __restrict__ p,
    const int* __restrict__ rulebook,
    int M,
    const T* __restrict__ W1,
    const T* __restrict__ b1,
    const T* __restrict__ W2,
    const T* __restrict__ b2,
    T* new_f,
    int* max_indices,
    const int C
) {
    __shared__ T shared_W1[32][C_in];
    __shared__ T shared_b1[32];         // Shared bias b1
    __shared__ T shared_W2[K][32];
    __shared__ T shared_b2[K];          // Shared bias b2
    __shared__ T shared_dp[K/V][V][C_in];
    __shared__ T shared_hidden[K/V][V][32];

    int block_output_start = blockIdx.y * K;
    int tx_local = threadIdx.x;
    int tx = tx_local + block_output_start;

    if (tx_local < 32){
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            shared_W1[tx_local][d] = W1[tx_local * C_in + d];
        }

        shared_b1[tx_local] = b1[tx_local];
    }

    #pragma unroll
    for (int d = 0; d < 32; d++) {
        shared_W2[tx_local][d] = W2[tx * 32 + d];
    }

    shared_b2[tx_local] = b2[tx_local];

    __syncthreads();

    int R0[V];
    int R1[V];
    int R2[V];

    int ty[V];

    #pragma unroll
    for (int v = 0; v < V; v++)
        ty[v] = threadIdx.y + v * (K / V);

    for (int s = blockIdx.x * K; s < M; s += K * gridDim.x) {
        #pragma unroll
        for (int v = 0; v < V; v++) {
            int rule_idx = 3 * (s + ty[v]);
            R0[v] = rulebook[rule_idx];
            R1[v] = rulebook[rule_idx + 1];
            R2[v] = rulebook[rule_idx + 2];
        }
        __syncthreads();

        #pragma unroll
        for (int v = 0; v < V; v++) {
            if (tx_local < C_in)
                shared_dp[threadIdx.y][v][tx_local] = p[R0[v] * C_in + tx_local] - p[R2[v] * C_in + tx_local];
        }
        __syncthreads();

        if (tx_local < 32){
            #pragma unroll
            for (int v = 0; v < V; v++) {
                shared_hidden[threadIdx.y][v][tx_local] = 0;
            }
        }
        __syncthreads();

        if (tx_local < 32){
            #pragma unroll
            for (int v = 0; v < V; v++) {

                #pragma unroll
                for (int d = 0; d < C_in; d++) {
                    shared_hidden[threadIdx.y][v][tx_local] += shared_W1[tx_local][d] * shared_dp[threadIdx.y][v][d];
                }
                shared_hidden[threadIdx.y][v][tx_local] += shared_b1[tx_local]; // Add bias b1
                // ReLU activation
                shared_hidden[threadIdx.y][v][tx_local] = fmaxf(0.0f, shared_hidden[threadIdx.y][v][tx_local]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int v = 0; v < V; v++) {

            T delta_f = 0;
            #pragma unroll
            for (int h = 0; h < 32; h++) {
                delta_f += shared_W2[tx_local][h] * shared_hidden[threadIdx.y][v][h];
            }
            delta_f += shared_b2[tx_local];

            T aggregated_f = f[R0[v] * C + tx] + delta_f;

            atomicMaxFloatIndex(&new_f[R1[v] * C + tx], &max_indices[R1[v] * C + tx], aggregated_f, s + ty[v]);
        }
        __syncthreads();
    }
}

template <typename T, int K, int V, int C_in>
__global__ void BiasMaxForwardKernelB(
    const T* __restrict__ f,             // [N, C]
    const T* __restrict__ p,             // [N, C_in]
    const int* __restrict__ rulebook,    // [M, 3]
    int M,
    const T* __restrict__ W1,            // [32, C_in]
    const T* __restrict__ b1,            // [32], new bias
    const T* __restrict__ W2,            // [C, 32]
    const T* __restrict__ b2,            // [C], new bias
    T* new_f,                            // [num_output, C]
    int* max_indices,                    // [num_output, C]
    int C,
    int o
) {
    // Define shared memory
    __shared__ T shared_W1[32][C_in];
    __shared__ T shared_b1[32];
    __shared__ T shared_W2[K][32];
    __shared__ T shared_b2[K];
    __shared__ T shared_dp[K/V][V][C_in];
    __shared__ T shared_hidden[K/V][V][32];

    int block_output_start = blockIdx.y * K;
    int tx_local = threadIdx.x;
    int tx = tx_local + block_output_start;

    if (tx_local < 32){
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            shared_W1[tx_local][d] = W1[tx_local * C_in + d];
        }
        shared_b1[tx_local] = b1[tx_local];
    }

    #pragma unroll
    for (int d = 0; d < 32; d++) {
        shared_W2[tx_local][d] = W2[tx * 32 + d];
    }

    shared_b2[tx_local] = b2[tx];

    __syncthreads();

    int R0[V];
    int R1[V];
    int R2[V];

    int ty[V];

    #pragma unroll
    for (int v = 0; v < V; v++)
        ty[v] = threadIdx.y + v * (K / V);

    for (int s = 0; s < M; s += K) {
        #pragma unroll
        for (int v = 0; v < V; v++) {
            if (s + ty[v] < M) {
                int rule_idx = 3 * (s + ty[v]);
                R0[v] = rulebook[rule_idx];
                R1[v] = rulebook[rule_idx + 1];
                R2[v] = rulebook[rule_idx + 2];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int v = 0; v < V; v++) {
            if (s + ty[v] < M) {
                if (tx_local < C_in)
                    shared_dp[threadIdx.y][v][tx_local] = p[R0[v] * C_in + tx_local] - p[R2[v] * C_in + tx_local];
            }
        }
        __syncthreads();

        if (tx_local < 32){
            #pragma unroll
            for (int v = 0; v < V; v++) {
                if (s + ty[v] < M) {
                    shared_hidden[threadIdx.y][v][tx_local] = 0;
                }
            }
        }
        __syncthreads();

        if (tx_local < 32){
            #pragma unroll
            for (int v = 0; v < V; v++) {
                if (s + ty[v] < M) {
                    #pragma unroll
                    for (int d = 0; d < C_in; d++) {
                        shared_hidden[threadIdx.y][v][tx_local] += shared_W1[tx_local][d] * shared_dp[threadIdx.y][v][d];
                    }
                    shared_hidden[threadIdx.y][v][tx_local] += shared_b1[tx_local]; // Add bias b1
                    // ReLU activation
                    shared_hidden[threadIdx.y][v][tx_local] = fmaxf(0.0f, shared_hidden[threadIdx.y][v][tx_local]);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int v = 0; v < V; v++) {
            if (s + ty[v] < M) {

                T delta_f = 0;
                #pragma unroll
                for (int h = 0; h < 32; h++) {
                    delta_f += shared_W2[tx_local][h] * shared_hidden[threadIdx.y][v][h];
                }
                delta_f += shared_b2[tx_local];

                T aggregated_f = f[R0[v] * C + tx] + delta_f;

                atomicMaxFloatIndex(&new_f[R1[v] * C + tx], &max_indices[R1[v] * C + tx], aggregated_f, o + s + ty[v]);
            }
        }
        __syncthreads();
    }
}

}

namespace {

template <typename T, int K, int V, int C_in>
__global__ void BiasMaxBackwardKernelA(
    const T* __restrict__ grad_output,          // [num_output, C]
    const int* __restrict__ max_indices,        // [num_output, C]
    const T* __restrict__ p,                    // [N, C_in]
    const int* __restrict__ rulebook,           // [M, 3]
    const T* __restrict__ W1,                   // [32, C_in]
    const T* __restrict__ b1,                   // [32], new bias
    const T* __restrict__ W2,                   // [C, 32]
    const T* __restrict__ b2,                   // [C], new bias
    T* __restrict__ grad_f,                     // [N, C]
    T* __restrict__ grad_W1,                    // [32, C_in]
    T* __restrict__ grad_b1,                    // [32], new bias gradient
    T* __restrict__ grad_W2,                    // [C, 32]
    T* __restrict__ grad_b2,                    // [C], new bias gradient
    int num_output,
    int C,
    int N
) {
    __shared__ T shared_W1[32][C_in];
    __shared__ T shared_b1[32];
    __shared__ T shared_W2[K][32];
    T temp_dp[V][C_in];
    T temp_hidden[V][32];

    T temp_grad_W1[32][C_in];
    T temp_grad_b1[32];
    T temp_grad_W2[32];
    T temp_grad_b2 = 0;

    int tx_local = threadIdx.x;
    int tx = tx_local + blockIdx.y * K;

    if (tx_local < 32){
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            shared_W1[tx_local][d] = W1[tx_local * C_in + d];
        }
        shared_b1[tx_local] = b1[tx_local];
    }

    #pragma unroll
    for (int h = 0; h < 32; h++) {
        shared_W2[tx_local][h] = W2[tx * 32 + h];
        temp_grad_W2[h] = 0.0f;
    }

    __syncthreads();

    // Initialize temporary gradients
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            temp_grad_W1[h][d] = 0.0f;
        }
        temp_grad_b1[h] = 0.0f;
    }
    __syncthreads();

    int ty[V];
    int output_idx[V];
    int R1[V];
    int R0[V];
    int R2[V];

    #pragma unroll
    for (int v = 0; v < V; v++)
        ty[v] = threadIdx.y + v * (K / V);

    for (int s = blockIdx.x * K; s < num_output; s += K * gridDim.x) {
        #pragma unroll
        for (int v = 0; v < V; v++) {
            output_idx[v] = s + ty[v];
            int input_idx = max_indices[output_idx[v] * C + tx];
            R0[v] = rulebook[3 * input_idx];
            R1[v] = rulebook[3 * input_idx + 1];
            R2[v] = rulebook[3 * input_idx + 2];
        }

        for (int v = 0; v < V; v++) {

            #pragma unroll
            for (int d = 0; d < C_in; d++) {
                temp_dp[v][d] = p[R0[v] * C_in + d] - p[R2[v] * C_in + d];
            }

            #pragma unroll
            for (int h = 0; h < 32; h++) {
                T hidden = 0.0f;
                #pragma unroll
                for (int d = 0; d < C_in; d++) {
                    hidden += shared_W1[h][d] * temp_dp[v][d];
                }
                hidden += shared_b1[h];

                hidden = fmaxf(0.0f, hidden);
                temp_hidden[v][h] = hidden;
            }
        }

        for (int v = 0; v < V; v++) {
            T grad = grad_output[output_idx[v] * C + tx];

            atomicAdd(&grad_f[R0[v] * C + tx], grad);

            #pragma unroll
            for (int h = 0; h < 32; h++) {
                T hidden = temp_hidden[v][h];
                T gw2 = grad * hidden;
                temp_grad_W2[h] += gw2;

                if (hidden > 0) {
                    T grad_hidden = grad * shared_W2[tx_local][h];
                    #pragma unroll
                    for (int d = 0; d < C_in; d++) {
                        T gw1 = grad_hidden * temp_dp[v][d];
                        temp_grad_W1[h][d] += gw1;
                    }
                    temp_grad_b1[h] += grad_hidden;
                }
            }
            temp_grad_b2 += grad;
        }
        __syncthreads();
    }
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < 32; h++) {
        atomicAdd(&grad_W2[tx * 32 + h], temp_grad_W2[h]);
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            atomicAdd(&grad_W1[h * C_in + d], temp_grad_W1[h][d]);
        }
        atomicAdd(&grad_b1[h], temp_grad_b1[h]);
    }
    atomicAdd(&grad_b2[tx], temp_grad_b2);
}

template <typename T, int K, int V, int C_in>
__global__ void BiasMaxBackwardKernelB(
    const T* __restrict__ grad_output,          // [num_output, C]
    const int* __restrict__ max_indices,        // [num_output, C]
    const T* __restrict__ p,                    // [N, C_in]
    const int* __restrict__ rulebook,           // [M, 3]
    const T* __restrict__ W1,                   // [32, C_in]
    const T* __restrict__ b1,                   // [32], new bias
    const T* __restrict__ W2,                   // [C, 32]
    const T* __restrict__ b2,                   // [C], new bias
    T* __restrict__ grad_f,                     // [N, C]
    T* __restrict__ grad_W1,                    // [32, C_in]
    T* __restrict__ grad_b1,                    // [32], new bias gradient
    T* __restrict__ grad_W2,                    // [C, 32]
    T* __restrict__ grad_b2,                    // [C], new bias gradient
    int num_output,
    int C,
    int N,
    int o
) {
    __shared__ T shared_W1[32][C_in];
    __shared__ T shared_b1[32];
    __shared__ T shared_W2[K][32];
    T temp_dp[V][C_in];
    T temp_hidden[V][32];

    T temp_grad_W1[32][C_in];
    T temp_grad_b1[32];
    T temp_grad_W2[32];
    T temp_grad_b2 = 0;

    int tx_local = threadIdx.x;
    int tx = tx_local + blockIdx.y * K;

    if (tx_local < 32){
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            shared_W1[tx_local][d] = W1[tx_local * C_in + d];
        }
        shared_b1[tx_local] = b1[tx_local];
    }

    #pragma unroll
    for (int h = 0; h < 32; h++) {
        shared_W2[tx_local][h] = W2[tx * 32 + h];
        temp_grad_W2[h] = 0.0f;
    }

    #pragma unroll
    for (int h = 0; h < 32; h++) {
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            temp_grad_W1[h][d] = 0.0f;
        }
        temp_grad_b1[h] = 0.0f;
    }
    __syncthreads();

    int ty[V];
    int output_idx[V];
    int R0[V];
    int R1[V];
    int R2[V];

    #pragma unroll
    for (int v = 0; v < V; v++)
        ty[v] = threadIdx.y + v * (K / V);

    for (int s = blockIdx.x * K; s < num_output; s += K * gridDim.x) {
        #pragma unroll
        for (int v = 0; v < V; v++) {
            output_idx[v] = s + ty[v];
            if (output_idx[v] < num_output) {

                int input_idx = max_indices[output_idx[v] * C + tx];
                R0[v] = rulebook[3 * input_idx];
                R1[v] = rulebook[3 * input_idx + 1];
                R2[v] = rulebook[3 * input_idx + 2];
            }
        }
        __syncthreads();

        for (int v = 0; v < V; v++) {
            if (output_idx[v] < num_output) {

                #pragma unroll
                for (int d = 0; d < C_in; d++) {
                    temp_dp[v][d] = p[R0[v] * C_in + d] - p[R2[v] * C_in + d];
                }

                #pragma unroll
                for (int h = 0; h < 32; h++) {
                    T hidden = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < C_in; d++) {
                        hidden += shared_W1[h][d] * temp_dp[v][d];
                    }
                    hidden += shared_b1[h];

                    hidden = fmaxf(0.0f, hidden);
                    temp_hidden[v][h] = hidden;
                }
            }
        }
        __syncthreads();

        for (int v = 0; v < V; v++) {
            if (output_idx[v] < num_output) {

                T grad = grad_output[output_idx[v] * C + tx];

                atomicAdd(&grad_f[R0[v] * C + tx], grad);

                #pragma unroll
                for (int h = 0; h < 32; h++) {
                    T hidden = temp_hidden[v][h];
                    T gw2 = grad * hidden;
                    temp_grad_W2[h] += gw2;

                    if (hidden > 0) {
                        T grad_hidden = grad * shared_W2[tx_local][h];
                        #pragma unroll
                        for (int d = 0; d < C_in; d++) {
                            T gw1 = grad_hidden * temp_dp[v][d];
                            temp_grad_W1[h][d] += gw1;
                        }
                        temp_grad_b1[h] += grad_hidden;
                    }
                }
                temp_grad_b2 += grad;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < 32; h++) {
        atomicAdd(&grad_W2[tx * 32 + h], temp_grad_W2[h]);
        #pragma unroll
        for (int d = 0; d < C_in; d++) {
            atomicAdd(&grad_W1[h * C_in + d], temp_grad_W1[h][d]);
        }
        atomicAdd(&grad_b1[h], temp_grad_b1[h]);
    }
    atomicAdd(&grad_b2[tx], temp_grad_b2);
}

}
std::tuple<torch::Tensor, torch::Tensor> bias_max_forward_cuda(
    torch::Tensor f,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {

    const int N = f.size(0);
    const int C = f.size(1);
    const int C_in = p.size(1);
    const int num_output = rulebook.select(1, 1).max().item<int>() + 1;

    auto options = f.options();
    torch::Tensor new_f = torch::full({num_output, C}, -FLT_MAX, options);
    torch::Tensor max_indices = torch::full({num_output, C}, -1, torch::dtype(torch::kInt32).device(options.device()));

    const float* d_f = f.data_ptr<float>();
    const float* d_p = p.data_ptr<float>();
    const int* d_rulebook = rulebook.data_ptr<int>();
    const float* d_W1 = W1.data_ptr<float>();
    const float* d_b1 = b1.data_ptr<float>();
    const float* d_W2 = W2.data_ptr<float>();
    const float* d_b2 = b2.data_ptr<float>();
    float* d_new_f = new_f.data_ptr<float>();
    int* d_max_indices = max_indices.data_ptr<int>();

    const int input_nPlanes = C;
    const int output_nPlanes = C;
    const int nHot = rulebook.size(0);

#define COO(CIN, TT, KK, VV) \
    else if ((input_nPlanes) % (KK) == 0 && (output_nPlanes) % (KK) == 0 && C_in == (CIN)) {  \
        int o = (nHot / (KK)) * (KK);                                   \
                                                                             \
        if (o >= (KK)) {                                                 \
            dim3 gridA(std::min(o / (KK), 512), (output_nPlanes) / (KK));    \
            dim3 blockA((KK), (KK) / (VV));                                    \
            BiasMaxForwardKernelA<TT, (KK), (VV), (CIN)><<<gridA, blockA>>>(      \
                d_f, d_p, d_rulebook, o, d_W1, d_b1, d_W2, d_b2, d_new_f, \
                d_max_indices, C                                         \
            );                                                           \
            cudaError_t errA = cudaGetLastError();                       \
            if (errA != cudaSuccess) {                                   \
                fprintf(stderr, "BiasMaxForwardKernelA launch failed: %s\n", \
                        cudaGetErrorString(errA));                       \
                exit(EXIT_FAILURE);                                      \
            }                                                            \
        }                                                                \
                                                                             \
        if ((nHot) > (o)) {                                              \
            dim3 gridB(1, (output_nPlanes) / (KK));                        \
            dim3 blockB((KK), (KK) / (VV));                                    \
            BiasMaxForwardKernelB<TT, (KK), (VV), (CIN)><<<gridB, blockB>>>(      \
                d_f, d_p, d_rulebook + 3 * o, nHot - o,                  \
                d_W1, d_b1, d_W2, d_b2, d_new_f, d_max_indices,          \
                C, o                                                       \
            );                                                           \
            cudaError_t errB = cudaGetLastError();                       \
            if (errB != cudaSuccess) {                                   \
                fprintf(stderr, "BiasMaxForwardKernelB launch failed: %s\n", \
                        cudaGetErrorString(errB));                       \
                exit(EXIT_FAILURE);                                      \
            }                                                            \
        }                                                                \
    }

#define FOO(TT, KK, VV)                                                 \
    if (false) {}                                                       \
    COO(3, TT, KK, VV)                                                  \
    COO(4, TT, KK, VV)                                                  \
    COO(5, TT, KK, VV)                                                  \
    COO(6, TT, KK, VV)                                                  \
    COO(7, TT, KK, VV)                                                  \
    COO(8, TT, KK, VV)                                                  \
    COO(9, TT, KK, VV)                                                  \
    COO(10, TT, KK, VV)                                                 \
    else {                                                              \
        fprintf(stderr, "Unsupported C_in value: %d\n", C_in);          \
        exit(EXIT_FAILURE);                                             \
    }

    // Invoke FOO macro
    FOO(float, 32, 32)

#undef COO
#undef FOO

    return std::make_tuple(new_f, max_indices);
}

std::vector<torch::Tensor> bias_max_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor max_indices,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {

    const int N = p.size(0);
    const int C = grad_output.size(1);
    const int C_in = p.size(1);
    const int num_output = grad_output.size(0);

    auto options = grad_output.options();
    torch::Tensor grad_f = torch::zeros({N, C}, options);
    torch::Tensor grad_W1 = torch::zeros({32, C_in}, options);
    torch::Tensor grad_b1 = torch::zeros({32}, options);
    torch::Tensor grad_W2 = torch::zeros({C, 32}, options);
    torch::Tensor grad_b2 = torch::zeros({C}, options);

    const float* d_grad_output = grad_output.data_ptr<float>();
    const int* d_max_indices = max_indices.data_ptr<int>();
    const float* d_p = p.data_ptr<float>();
    const int* d_rulebook = rulebook.data_ptr<int>();
    const float* d_W1 = W1.data_ptr<float>();
    const float* d_b1 = b1.data_ptr<float>();
    const float* d_W2 = W2.data_ptr<float>();
    const float* d_b2 = b2.data_ptr<float>();
    float* d_grad_f = grad_f.data_ptr<float>();
    float* d_grad_W1 = grad_W1.data_ptr<float>();
    float* d_grad_b1 = grad_b1.data_ptr<float>();
    float* d_grad_W2 = grad_W2.data_ptr<float>();
    float* d_grad_b2 = grad_b2.data_ptr<float>();

    const int input_nPlanes = C;
    const int output_nPlanes = C;

#define COO(CIN, TT, KK, VV) \
    else if ((input_nPlanes) % (KK) == 0 && (output_nPlanes) % (KK) == 0 && C_in == (CIN)) {  \
        int o = (num_output / (KK)) * (KK);                             \
                                                                             \
        if (o >= (KK)) {                                                 \
            dim3 gridA(std::min(o / (KK), 512), (output_nPlanes) / (KK));    \
            dim3 blockA((KK), (KK) / (VV));                                    \
            BiasMaxBackwardKernelA<TT, (KK), (VV), (CIN)><<<gridA, blockA>>>(    \
                d_grad_output, d_max_indices, d_p, d_rulebook, d_W1, d_b1, \
                d_W2, d_b2, d_grad_f, d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2, \
                o, C, N                                                  \
            );                                                           \
            cudaError_t errA = cudaGetLastError();                       \
            if (errA != cudaSuccess) {                                   \
                fprintf(stderr, "BiasMaxBackwardKernelA launch failed: %s\n", \
                        cudaGetErrorString(errA));                       \
                exit(EXIT_FAILURE);                                      \
            }                                                            \
        }                                                                \
                                                                             \
        if ((num_output) > (o)) {                                        \
            dim3 gridB(1, (output_nPlanes) / (KK));                        \
            dim3 blockB((KK), (KK) / (VV));                                    \
            BiasMaxBackwardKernelB<TT, (KK), (VV), (CIN)><<<gridB, blockB>>>(    \
                d_grad_output + o * C, d_max_indices + o * C,            \
                d_p, d_rulebook, d_W1, d_b1, d_W2, d_b2,                 \
                d_grad_f, d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,    \
                num_output - o, C, N, o                                  \
            );                                                           \
            cudaError_t errB = cudaGetLastError();                       \
            if (errB != cudaSuccess) {                                   \
                fprintf(stderr, "BiasMaxBackwardKernelB launch failed: %s\n", \
                        cudaGetErrorString(errB));                       \
                exit(EXIT_FAILURE);                                      \
            }                                                            \
        }                                                                \
    }

#define FOO(TT, KK, VV)                                                 \
    if (false) {}                                                       \
    COO(3, TT, KK, VV)                                                  \
    COO(4, TT, KK, VV)                                                  \
    COO(5, TT, KK, VV)                                                  \
    COO(6, TT, KK, VV)                                                  \
    COO(7, TT, KK, VV)                                                  \
    COO(8, TT, KK, VV)                                                  \
    COO(9, TT, KK, VV)                                                  \
    COO(10, TT, KK, VV)                                                 \
    else {                                                              \
        fprintf(stderr, "Unsupported C_in value: %d\n", C_in);          \
        exit(EXIT_FAILURE);                                             \
    }

    // Invoke FOO macro
    FOO(float, 32, 32)

#undef COO
#undef FOO

    return {grad_f, grad_W1, grad_b1, grad_W2, grad_b2};
}