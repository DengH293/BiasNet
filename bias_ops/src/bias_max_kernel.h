#ifndef BIAS_MAX_KERNEL_H
#define BIAS_MAX_KERNEL_H

#include <torch/extension.h>
#include <vector>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)


std::tuple<torch::Tensor, torch::Tensor> bias_max_forward(
    torch::Tensor f,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
);


std::vector<torch::Tensor> bias_max_backward(
    torch::Tensor grad_output,
    torch::Tensor max_indices,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
);


std::tuple<torch::Tensor, torch::Tensor> bias_max_forward_cuda(
    torch::Tensor f,          // [N, C]
    torch::Tensor p,          // [N, 3]
    torch::Tensor rulebook,   // [M, 3]
    torch::Tensor W1,         // [32, C_in]
    torch::Tensor b1,         // [32]
    torch::Tensor W2,         // [C, 32]
    torch::Tensor b2          // [32]
);

std::vector<torch::Tensor> bias_max_backward_cuda(
    torch::Tensor grad_output,     // [num_output, C]
    torch::Tensor max_indices,     // [num_output, C]
    torch::Tensor p,               // [N, 3]
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
);

#endif // BIAS_MAX_KERNEL_H
