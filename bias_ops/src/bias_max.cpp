#include <torch/extension.h>
#include <vector>
#include "bias_max_kernel.h"


std::tuple<torch::Tensor, torch::Tensor> bias_max_forward(
    torch::Tensor f,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {
    return bias_max_forward_cuda(f, p, rulebook, W1, b1, W2, b2);
}


std::vector<torch::Tensor> bias_max_backward(
    torch::Tensor grad_output,
    torch::Tensor max_indices,
    torch::Tensor p,
    torch::Tensor rulebook,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {
    return bias_max_backward_cuda(grad_output, max_indices, p, rulebook, W1, b1, W2, b2);
}
