#pragma once
#include <stdint.h>

#define M 2
#define K 2
#define N 2

/* 行主序(row-major)展平存储 */
__attribute__((section(".matA"), aligned(64))) static const int8_t A[M*K] = {
  12, -7,
  -34, 56
};

__attribute__((section(".matB"), aligned(64))) static const int8_t B[K*N] = {
  3, -2,
  5,  4
};


/* C 初始缓冲（随机生成，用作输入/初始累加器） */
__attribute__((section(".matC"), aligned(64))) static int32_t C[4*4] = {
  0, 0,1,1,
  0, 0,1,1,
  1, 1,1,1,
  1, 1,1,1
};


/* D 为 A*BT 的 golden 结果（int32 累加） */
__attribute__((aligned(64))) static const int32_t D_ref[4*4] = {
50,32,1,1,
-214,54,1,1,
1,1,1,1,
1,1,1,1
};
