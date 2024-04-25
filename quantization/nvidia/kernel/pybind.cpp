#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "gemm_quant.h"

PYBIND11_MODULE(quant, m){
  m.def("gemm_awq_ut", &gemm_awq_ut);
  m.def("transpose_merge_zeros_scales", &transpose_merge_zeros_scales);
  m.def("convert_ours_to_awq", &convert_ours_to_awq);
  m.def("convert_awq_to_lmdeploy", &convert_awq_to_lmdeploy);
}
