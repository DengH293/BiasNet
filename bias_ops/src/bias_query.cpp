#include <torch/extension.h>
#include "bias_query_kernel.h"

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> generate_hash_table_wrapper(torch::Tensor key) {
    // Validate input tensor
    CHECK_INPUT(key);
    TORCH_CHECK(key.dim() == 1, "key must be a 1D tensor");

    const int n = key.size(0);

    // Calculate the next power of two for hash table size
    auto next_power_of_two = [](int64_t n) -> int64_t {
        if (n <= 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    };

    int64_t HASH_SIZE = next_power_of_two(static_cast<int64_t>(n) * 2); // Multiply by 2 for lower load factor

    // Allocate hash table tensors
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(key.device());
    auto hash_table_keys = torch::full({HASH_SIZE}, EMPTY_KEY, options);
    auto hash_table_values = torch::zeros({HASH_SIZE}, options);

    // Access pointers
    int64_t* d_hash_table_keys = hash_table_keys.data_ptr<int64_t>();
    int64_t* d_hash_table_values = hash_table_values.data_ptr<int64_t>();
    const int64_t* d_keys = key.data_ptr<int64_t>();

    // Insert keys into the hash table
    hash_insert_cuda_launcher(n, d_keys, d_hash_table_keys, d_hash_table_values, HASH_SIZE);

    return {hash_table_keys, hash_table_values};
}

std::vector<torch::Tensor> compute_counts_wrapper(
    int kernel_size,
    torch::Tensor new_xyz,
    torch::Tensor hash_table_keys,
    torch::Tensor hash_table_values
) {
    // Validate input tensors
    CHECK_INPUT(new_xyz);
    CHECK_INPUT(hash_table_keys);
    CHECK_INPUT(hash_table_values);

    const int m = new_xyz.size(0);
    TORCH_CHECK(new_xyz.dim() == 2 && new_xyz.size(1) == 4, "new_xyz must be a 2D tensor with shape (m, 4)");

    int64_t HASH_SIZE = hash_table_keys.size(0);

    // Allocate counts tensor
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(new_xyz.device());
    auto counts = torch::zeros({m}, options);

    // Access pointers
    const int64_t* d_new_xyz = new_xyz.data_ptr<int64_t>();
    int64_t* d_counts = counts.data_ptr<int64_t>();
    const int64_t* d_hash_table_keys = hash_table_keys.data_ptr<int64_t>();
    const int64_t* d_hash_table_values = hash_table_values.data_ptr<int64_t>();

    // Compute counts using the CUDA kernel
    compute_counts_cuda_launcher(m, kernel_size, d_hash_table_keys, d_hash_table_values, HASH_SIZE, d_new_xyz, d_counts);

    return {counts};
}

std::vector<torch::Tensor> fill_neighbor_indices_wrapper(
    int kernel_size,
    torch::Tensor new_xyz,
    torch::Tensor hash_table_keys,
    torch::Tensor hash_table_values,
    torch::Tensor counts,
    torch::Tensor offsets
) {
    // Validate input tensors
    CHECK_INPUT(new_xyz);
    CHECK_INPUT(hash_table_keys);
    CHECK_INPUT(hash_table_values);
    CHECK_INPUT(counts);
    CHECK_INPUT(offsets);

    const int m = new_xyz.size(0);

    TORCH_CHECK(new_xyz.dim() == 2 && new_xyz.size(1) == 4, "new_xyz must be a 2D tensor with shape (m, 4)");
    TORCH_CHECK(counts.dim() == 1 && counts.size(0) == m, "counts must be a 1D tensor of length m");
    TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) == m + 1, "offsets must be a 1D tensor of length m + 1");

    int64_t HASH_SIZE = hash_table_keys.size(0);

    // Determine total number of neighbors
    int64_t total_neighbors = offsets[offsets.size(0) - 1].item<int64_t>();

    // Allocate tensors for input and output indices
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(new_xyz.device());
    auto input_idx = torch::empty({total_neighbors}, options);
    auto output_idx = torch::empty({total_neighbors}, options);

    // Access pointers
    const int64_t* d_new_xyz = new_xyz.data_ptr<int64_t>();
    int64_t* d_input_idx = input_idx.data_ptr<int64_t>();
    int64_t* d_output_idx = output_idx.data_ptr<int64_t>();
    int64_t* d_counts = counts.data_ptr<int64_t>();
    int64_t* d_offsets = offsets.data_ptr<int64_t>();
    const int64_t* d_hash_table_keys = hash_table_keys.data_ptr<int64_t>();
    const int64_t* d_hash_table_values = hash_table_values.data_ptr<int64_t>();

    // Fill neighbor indices using the CUDA kernel
    fill_neighbor_indices_cuda_launcher(
        m, kernel_size, d_hash_table_keys, d_hash_table_values, HASH_SIZE,
        d_new_xyz, d_input_idx, d_output_idx, d_counts, d_offsets
    );

    return {input_idx, output_idx};
}
