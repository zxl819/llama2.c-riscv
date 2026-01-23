#pragma once
#include <stdint.h>

// FP8 E4M3FN (1 sign, 4 exponent bits, 3 mantissa bits), bias=7.
// "FN" indicates no Inf; exponent=0xF is treated as NaN here.
// This header provides a lightweight decoder to float for testing.
// It is intentionally freestanding-friendly (no libm required).

static inline float fp8e4m3fn_to_f32(uint8_t x) {
    const uint32_t sign = (uint32_t)(x >> 7);
    const uint32_t exp  = (uint32_t)((x >> 3) & 0x0F);
    const uint32_t man  = (uint32_t)(x & 0x07);

    // NaN (E=0xF). For FN, treat as NaN.
    if (exp == 0x0F) {
        // Quiet NaN: 0x7FC00000
        union { uint32_t u; float f; } v;
        v.u = 0x7FC00000u;
        return v.f;
    }

    // Zero / subnormal
    if (exp == 0) {
        if (man == 0) {
            union { uint32_t u; float f; } v;
            v.u = sign ? 0x80000000u : 0u;
            return v.f;
        }

        // subnormal: value = (-1)^sign * 2^(1-bias) * (man / 2^3)
        // bias=7 => 2^(-6) * (man/8) = man * 2^(-9)
        float v = (float)man;
        // scale by 2^-9
        for (int i = 0; i < 9; ++i) v *= 0.5f;
        return sign ? -v : v;
    }

    // normal: (-1)^sign * 2^(exp-bias) * (1 + man/8)
    // Build float bits directly for exactness.
    const int32_t unbiased = (int32_t)exp - 7; // bias=7
    const uint32_t fexp = (uint32_t)(unbiased + 127);
    const uint32_t fman = man << (23 - 3);

    union { uint32_t u; float f; } v;
    v.u = (sign << 31) | (fexp << 23) | fman;
    return v.f;
}

// Convenience: convert row-major fp8 matrix to float matrix.
static inline void fp8e4m3fn_mat_to_f32(float *dst, const uint8_t *src, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        dst[i] = fp8e4m3fn_to_f32(src[i]);
    }
}
