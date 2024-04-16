

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/block_sparse_attention_lut.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention_prefill_warp", &sparse_attention_prefill_warp);
    m.def("sparse_attention_prefill", &sparse_attention_prefill);
    m.def("add", &add);
}

