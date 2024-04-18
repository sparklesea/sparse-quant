#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "gemm_quant.h"

PYBIND11_MODULE(quant, m){
  m.def("gemm_awq_ut", &gemm_awq_ut);
  m.def("test_wmma", &test_wmma);
}
