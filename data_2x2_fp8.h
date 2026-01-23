#pragma once
#include <stdint.h>

#define M 2
#define K 2
#define N 2

// FP8(E4M3FN) 测试数据（字节存储）。
// 下面的编码在 fp8_e4m3fn.h 中有对应解码：
//   1.0  -> 0x38
//   1.5  -> 0x3C
//   2.0  -> 0x40
//  -0.5  -> 0xB0
//  -1.25 -> 0xBA
//
// 本测试例使用 Matrix 的 mfqma.cf 路径：输入为 fp8 字节，累加/输出为 fp32。

/* 行主序(row-major)展平存储 */
__attribute__((section(".matA"), aligned(64))) static const uint8_t A_fp8[M*K] = {
  0x3C, 0xB0, //  1.5, -0.5
  0xBA, 0x40  // -1.25, 2.0
};

__attribute__((section(".matB"), aligned(64))) static const uint8_t B_fp8[K*N] = {
  0x40, 0x38, // 2.0, 1.0
  0x3C, 0xB0  // 1.5, -0.5
};


/* C 初始缓冲（fp32，作为累加器初值） */
__attribute__((section(".matC"), aligned(64))) static float C[4*4] = {
  0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f,
};


/* D_ref 为 A*B 的 golden 结果（fp32） */
__attribute__((aligned(64))) static const float D_ref[4*4] = {
  // A = [ 1.5, -0.5; -1.25, 2.0 ]
  // B = [ 2.0,  1.0;  1.5, -0.5 ]
  // C00=2.25, C01=1.75, C10=0.5, C11=-2.25
  2.25f,  1.75f, 0.0f, 0.0f,
  0.5f,  -2.25f, 0.0f, 0.0f,
  0.0f,   0.0f,  0.0f, 0.0f,
  0.0f,   0.0f,  0.0f, 0.0f,
};
