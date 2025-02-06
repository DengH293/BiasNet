// bias_query_kernel.h

#ifndef BIAS_QUERY_KERNEL_H
#define BIAS_QUERY_KERNEL_H

#include <stdint.h>
#include <vector>
#include <torch/extension.h>

#ifndef EMPTY_KEY
#define EMPTY_KEY (-1)
#endif

void hash_insert_cuda_launcher(int n, const int64_t* d_keys,
                               int64_t* d_hash_table_keys,
                               int64_t* d_hash_table_values,
                               int64_t HASH_SIZE);

void compute_counts_cuda_launcher(int m, int kernel_size,
                                  const int64_t* d_hash_table_keys,
                                  const int64_t* d_hash_table_values,
                                  int64_t HASH_SIZE,
                                  const int64_t* new_xyz,
                                  int64_t* counts);

void fill_neighbor_indices_cuda_launcher(int m, int kernel_size,
                                         const int64_t* d_hash_table_keys,
                                         const int64_t* d_hash_table_values,
                                         int64_t HASH_SIZE,
                                         const int64_t* new_xyz,
                                         int64_t* input_idx, int64_t* output_idx,
                                         const int64_t* counts, const int64_t* offsets);

std::vector<torch::Tensor> generate_hash_table_wrapper(torch::Tensor key);

std::vector<torch::Tensor> compute_counts_wrapper(int kernel_size,
                                                  torch::Tensor new_xyz,
                                                  torch::Tensor hash_table_keys,
                                                  torch::Tensor hash_table_values);

std::vector<torch::Tensor> fill_neighbor_indices_wrapper(int kernel_size,
                                                         torch::Tensor new_xyz,
                                                         torch::Tensor hash_table_keys,
                                                         torch::Tensor hash_table_values,
                                                         torch::Tensor counts,
                                                         torch::Tensor offsets);

#endif // BIAS_QUERY_KERNEL_H
