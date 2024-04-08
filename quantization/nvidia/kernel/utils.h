#pragma once

#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype) TORCH_CHECK(x.dtype() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim) TORCH_CHECK(x.dim() == true_dim, "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_NUMEL(x, minimum) TORCH_CHECK(x.numel() >= minimum, "Tensor " #x " must have at last " #minimum " elements")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) TORCH_CHECK(x.stride(-1) == 1, "Tensor " #x " must be contiguous at the last dimension")

// Define rope type
enum ROPE_TYPE {
  NO_ROPE = 0,    // do not use rope
  FULL_ROPE = 1,  // use rope for all headdim
  HALF_ROPE = 2   // use rope for half of headdim
};

// Define which dim the sequence length is
enum SEQ_DIM_TYPE {
  FIRST = 0,   // [seqlen, ...]
  SECOND = 1,  // [..., seqlen, ...]
};

// Define attention mask type
enum MASK_TYPE {
  NO_MASK = 0,    // do not use mask
  ALIBI_MASK = 1  // use alibi mask
};

// Define whether the token index is aligned for different token
enum FREQ_ALIGNED {
  NO = 0,
  YES = 1
};