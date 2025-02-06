#include <torch/serialize/tensor.h>
#include <torch/extension.h>


#include "bias_query_kernel.h"
#include "bias_max_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_hash_table", &generate_hash_table_wrapper, "Point Hash CUDA (CUDA)");
    m.def("compute_counts", &compute_counts_wrapper, "Point Hash CUDA (CUDA)");
    m.def("fill_neighbor_indices", &fill_neighbor_indices_wrapper, "Point Hash CUDA (CUDA)");
    m.def("bias_max_forward", &bias_max_forward, "Bias Max Forward CUDA");
    m.def("bias_max_backward", &bias_max_backward, "Bias Max Backward CUDA");
}
