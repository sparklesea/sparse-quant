// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "metric.h"
#include "macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include <torch/extension.h>

namespace turbomind {

extern bool g_dump_kernel_info_once;

class GemmS4F16 {
   public:
    GemmS4F16();

    ~GemmS4F16();

    enum Type {
        kGemm,
        kFusedSiluFfn
    };

    void Measure(half* C,
                 const uint* A,
                 const half* B,
                 const half2* Q,
                 int m,
                 int n,
                 int k,
                 int group_size,
                 Type type,
                 std::vector<Metric>& metrics,
                 cudaStream_t st);

    void Run(half* C,
             const uint* A,
             const half* B,
             const half2* Q,
             int m,
             int n,
             int k,
             int group_size,
             Type type,
             int algo_id);  // cudaStream_t st

    void RunWrapper(torch::Tensor input,
                    torch::Tensor qweight,
                    torch::Tensor zeros_scales,
                    torch::Tensor output,
                    const int batch_size,
                    const int input_features,
                    const int output_features,
                    const int group_size);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
