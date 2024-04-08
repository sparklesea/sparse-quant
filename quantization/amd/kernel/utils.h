#ifndef _UTILS_H_
#define _UTILS_H_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

#include "../utils.h"

#define WARP_SIZE 64
#define WARP_SIZE_HALF 32
#define FLOAT_BANK_SIZE 32
#define MAX_HEAD_SIZE 128
#define MAX_LEN_GROUP 64
#define MAX_LOOP_SPACE 2

#define MAX_THREADS_PER_BLOCK 1024

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {

  if (threadNum >= 64)
    sum += __shfl_down(sum, 32, WARP_SIZE); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 32)
    sum += __shfl_down(sum, 16, WARP_SIZE); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down(sum, 8, WARP_SIZE); // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down(sum, 4, WARP_SIZE); // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down(sum, 2, WARP_SIZE); // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down(sum, 1, WARP_SIZE); // 0-1, 2-3, 4-5, etc.
  return sum;
}

__device__ __forceinline__ float warpReduceMax(float max_val,
                                               unsigned int threadNum) {

  if (threadNum >= 64)
    max_val = max(
        max_val, __shfl_down(max_val, 32, WARP_SIZE)); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 32)
    max_val = max(
        max_val, __shfl_down(max_val, 16, WARP_SIZE)); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    max_val = max(max_val,
                  __shfl_down(max_val, 8, WARP_SIZE)); // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    max_val =
        max(max_val, __shfl_down(max_val, 4, WARP_SIZE)); // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    max_val = max(max_val, __shfl_down(max_val, 2,
                                       WARP_SIZE)); // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    max_val =
        max(max_val, __shfl_down(max_val, 1, WARP_SIZE)); // 0-1, 2-3, 4-5, etc.
  return max_val;
}

__device__ float blockReduceSum_blockdimx_128(float reducing,
                                              float *shared_mem) {
  // const int32_t WPT = blockDim.x / WARP_SIZE > WARP_SIZE_HALF ? WARP_SIZE :
  // (blockDim.x / WARP_SIZE);
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t warp_id = threadIdx.x / WARP_SIZE;

  reducing += __shfl_xor(reducing, 32, WARP_SIZE);
  reducing += __shfl_xor(reducing, 16, WARP_SIZE);
  reducing += __shfl_xor(reducing, 8, WARP_SIZE);
  reducing += __shfl_xor(reducing, 4, WARP_SIZE);
  reducing += __shfl_xor(reducing, 2, WARP_SIZE);
  reducing += __shfl_xor(reducing, 1, WARP_SIZE);

  if (lane_id == 0)
    shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < 2)
    reducing = shared_mem[lane_id];

  reducing += __shfl_xor(reducing, 1, WARP_SIZE);

  reducing = __shfl(reducing, 0, WARP_SIZE);
  return reducing;
}

__device__ float blockReduceSum(float reducing, float *shared_mem) {
  // Helper function for reduce softmax exp sum.
  const int32_t WPT = blockDim.x / WARP_SIZE > WARP_SIZE_HALF
                          ? WARP_SIZE
                          : (blockDim.x / WARP_SIZE);
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t warp_id = threadIdx.x / WARP_SIZE;

  // # pragma unroll
  for (int32_t mask = WARP_SIZE_HALF; mask >= 1; mask /= 2) {
    reducing += __shfl_xor(reducing, mask, WARP_SIZE);
  }

  if (lane_id == 0)
    shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < WPT)
    reducing = shared_mem[lane_id];

  // # pragma unroll
  for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
    reducing += __shfl_xor(reducing, mask, WARP_SIZE);
  }
  reducing = __shfl(reducing, 0, WARP_SIZE);
  return reducing;
}

__device__ __forceinline__ float blockReduceSum_wp32(float reducing,
                                                     float *shared_mem) {
  // Helper function for reduce softmax exp sum.
  const int32_t WPT = blockDim.x / 32;
  int32_t WPTB = WPT == 20 ? 32 : WPT;
  const int32_t lane_id = threadIdx.x % 32;
  const int32_t warp_id = threadIdx.x / 32;

#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask /= 2) {
    reducing += __shfl_xor(reducing, mask, 32);
  }

  if (lane_id == 0)
    shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < WPTB)
    reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

#pragma unroll
  for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
    reducing += __shfl_xor(reducing, mask, 32);
  }
  reducing = __shfl(reducing, 0, 32);
  return reducing;
}

// reference: https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/gemm_s_f16/common.h#L115
__inline__ __device__ uint4 dequantize_s4_to_fp16x2_v2(uint32_t const& source) {
    uint4 result;  // 4 * sizeof(uint) == 16Bytes ---> 8 half elements

    // source, uint32_t ---> 8 uint4b_t elements ---> dequant to 8 half elements
    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
    static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4, 0x05400540

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.

    // source  : X7 X6 X5 X4 X3 X2 X1 X0
    // top_i4s :  0  0 X7 X6 X5 X4 X3 X2
    const uint32_t top_i4s = i4s >> 8;

    {
        //  64 only, trade 4 hfma2 with 2 shifts
        // source  : X7 X6 X5 X4 X3 X2 X1 X0
        // top_i4s :  0  0 X7 X6 X5 X4 X3 X2

        // 0x 0-0-0-X4-0-0-0-X0, 0x(0-5-4-X4-0-5-4-X0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        h[0] = (i4s & BOT_MASK) | MAGIC_NUM_2;

        // 0x 0-0-X5-0-0-0-X1-0, 0x(5-4-X5-0-5-4-X1-0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[1] = (i4s & TOP_MASK) | MAGIC_NUM_1;

        // 0x 0-0-0-X6-0-0-0-X2, 0x(0-5-4-X6-0-5-4-X2)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        h[2] = (top_i4s & BOT_MASK) | MAGIC_NUM_2;

        // 0x 0-0-X7-0-0-0-X3-0, 0x(5-4-X7-0-5-4-X3-0)
        // asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[3] = (top_i4s & TOP_MASK) | MAGIC_NUM_1;

        h[0] <<= 4;  // 0x 5-4-X4-0-5-4-X0-0, h[1]: 0x 5-4-X5-0-5-4-X1-0
        h[2] <<= 4;  // 0x 5-4-X6-0-5-4-X2-0, h[3]: 0x 5-4-X7-0-5-4-X3-0
        // X4 : [0, 15]
        // [64, 79] <---> [0, 15] <---> uint4b_t

        // we don't need to subtract the magic nums because zeros will go through the same dequant function
        // and carry the same magic constant, the magic num will be canceled out after subtracting zeros
    }

    return result;
}


#endif
