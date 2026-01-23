#pragma once
#include <stdint.h>

// 2x2 FP8(E4M3FN) test data.
// Storage is row-major flattened.
//
// FP8 E4M3FN encoding quick cheatsheet (bias=7):
//   1.0  -> 0x38
//   1.5  -> 0x3C
//   2.0  -> 0x40
//  -0.5  -> 0xB0
//  -1.25 -> 0xBA

#define M 2
#define K 2
#define N 2

// A: [M x K] fp8 bytes (row-major)
__attribute__((section(".matA"), aligned(64))) static const uint8_t A_fp8[M*K] = {
    0x3C, 0xB0, //  1.5, -0.5
    0xBA, 0x40  // -1.25, 2.0
};

// B: [K x N] fp8 bytes (row-major)
__attribute__((section(".matB"), aligned(64))) static const uint8_t B_fp8[K*N] = {
    0x40, 0x38, // 2.0, 1.0
    0x3C, 0xB0  // 1.5, -0.5
};

// Golden reference in float32 for C = A * B.
// C(0,0) = 1.5*2.0 + (-0.5)*1.5 = 2.25
// C(0,1) = 1.5*1.0 + (-0.5)*(-0.5) = 1.75
// C(1,0) = (-1.25)*2.0 + 2.0*1.5 = 0.5
// C(1,1) = (-1.25)*1.0 + 2.0*(-0.5) = -2.25
__attribute__((aligned(64))) static const float C_ref[M*N] = {
    2.25f,  1.75f,
    0.50f, -2.25f,
};
