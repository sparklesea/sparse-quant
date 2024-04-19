

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/block_sparse_attention_lut.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add);
    m.def("sparse_attention_prefill_p", &sparse_attention_prefill_p);
    m.def("sparse_attention_prefill_p_64", &sparse_attention_prefill_p_64);
}

