/*
---
title: Softmax RVV 正确性验证测试
date: 2026-01-29
description: 用于验证从 llama2.c-riscv 提取的 Softmax RVV 算子的正确性，包含参考实现对比与 MaxDiff 统计。
author: zhaoxinlei
version: 1.0
---
============================================================================================================================================================
代码说明
============================================================================================================================================================
该测试程序提供了一个完整的回归环境：
1. 集成了外部定义的 softmax_stable_rvv_fp32 算子。
2. 内部实现了一个基于标量的 softmax_ref 作为 Golden 参考（包含简单的 expf 逻辑）。
3. 使用 xorshift64* 算法生成稳定的随机测试激励，输入范围约为 [-3, 3]。
4. 处理了异常捕获（handle_trap），当执行非法 RVV 指令时会通过 UART 打印诊断信息。
5. 结果校验：计算 MaxDiff (L-inf norm)，容差设置为 1e-3，以允许向量多项式与标量多项式之间的微小误差。
============================================================================================================================================================
*/

// Minimal bare-metal correctness test for softmax_rvv.c

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "uart_helper.c"

#define CLOCK_FREQUENCY 24000000
#define UART_BITRATE    115200

// Declare the external RVV function
void softmax_stable_rvv_fp32(float* dst, const float* src, size_t n);

// ---------------------------------------------------------------------------
// Tiny libc hooks provided by bare_syscalls_uart.c
void *memcpy(void *dest, const void *src, size_t len);
void *memset(void *dest, int byte, size_t len);
void exit(int code);

// ---------------------------------------------------------------------------
// Trap diagnostics
static inline uintptr_t read_csr_mtval(void) {
    uintptr_t x;
    __asm__ volatile ("csrr %0, mtval" : "=r"(x));
    return x;
}

static inline void enable_rvv_state(void) {
    uintptr_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9);   // VS=Dirty
    mstatus |= (uintptr_t)(3u << 13);  // FS=Dirty
    __asm__ volatile ("csrw mstatus, %0" :: "r"(mstatus) : "memory");
}

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
    (void)regs;
    uintptr_t mtval = read_csr_mtval();
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("\r\n[test trap] mcause=0x");
    print_uart_hex((unsigned long long)cause);
    print_uart(" mepc=0x");
    print_uart_hex((unsigned long long)epc);
    print_uart(" mtval=0x");
    print_uart_hex((unsigned long long)mtval);
    print_uart("\r\n");
    exit(224);
    return epc;
}

// ---------------------------------------------------------------------------
// Math helpers for Reference Implementation

static float bare_fabsf(float x) { return x < 0.0f ? -x : x; }

// Simple expf approximation for reference (matches run.c)
float expf(float x) {
    const float LOG2E = 1.4426950408889634f;
    float y = x * LOG2E;
    float ip = y >= 0.0f ? (float)((int)(y + 0.5f)) : (float)((int)(y - 0.5f));
    float fp = y - ip;
    float poly = 1.0f + fp * (0.696065642f + fp * 0.224494337f);
    int exponent = (int)ip + 127;
    if (exponent <= 0) exponent = 0;
    if (exponent >= 255) exponent = 255;
    union { uint32_t i; float f; } bits;
    bits.i = (uint32_t)(exponent << 23);
    return bits.f * poly;
}

void softmax_ref(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

// ---------------------------------------------------------------------------
// PRNG
static uint64_t g_rng = 0x123456789abcdef0ull;
static inline uint32_t rng_u32(void) {
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1Dull) >> 32);
}

static inline float rng_f32_signed(void) {
    uint32_t u = rng_u32();
    float f = (float)(u >> 8) * (1.0f / 16777216.0f); // [0,1)
    return (f * 2.0f) - 1.0f;
}

// ---------------------------------------------------------------------------
// Main Test

#define MAX_N 1024
static float input_buf[MAX_N] __attribute__((aligned(64)));
static float ref_buf[MAX_N] __attribute__((aligned(64)));
static float rvv_buf[MAX_N] __attribute__((aligned(64)));

int main(void) {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("\r\n=== Test Softmax RVV ===\r\n");
    enable_rvv_state();

    int sizes[] = {1024};
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    int errors = 0;

    for (int k = 0; k < num_sizes; k++) {
        int n = sizes[k];
        print_uart("Testing N="); print_uart_int_dec(n); print_uart("... ");

        // Init Data
        for (int i = 0; i < n; i++) {
            input_buf[i] = rng_f32_signed() * 3.0f; // Scale a bit
        }

        // Run Reference
        for(int i=0; i<n; ++i) ref_buf[i] = input_buf[i];
        softmax_ref(ref_buf, n);

        // Run RVV
        // Softmax is often in-place or safe out-of-place? 
        // RVV signature is (dst, src, n). Let's test out-of-place first if supported, 
        // but typically softmax is used in place. 
        // The implementation I saw: 
        //   softmax_stable_rvv_fp32(float* dst, const float *src, size_t n)
        // If src==dst it should work (pointers are just copied).
        // Let's test out-of-place correctness.
        softmax_stable_rvv_fp32(rvv_buf, input_buf, n);

        print_uart("Head comparison (val*1e6):\r\n");
        for (int i=0; i<n && i<16; ++i) {
             print_uart("  i="); print_uart_int_dec(i);
             print_uart(" ref="); print_float_fixed3(ref_buf[i]);
             print_uart(" rvv="); print_float_fixed3(rvv_buf[i]);
             print_uart(" diff="); 
             float d = rvv_buf[i] - ref_buf[i];
             print_float_fixed3(d);
             print_uart("\r\n");
        }

        // Compare
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = bare_fabsf(rvv_buf[i] - ref_buf[i]);
            if (diff > max_diff) max_diff = diff;
        }

        print_uart("MaxDiff="); 
        // print float manually-ish
        int d_int = (int)(max_diff * 1000000.0f);
        print_uart_int_dec(d_int); print_uart("e-6 ");

        // Tolerance: exp approximation differences can accumulate. 
        // 1e-4 is usually generous enough for simple exp comparisons, 
        // but here we have vector exp poly vs scalar exp poly. 
        // They might not match bit-exact.
        if (max_diff > 1e-3f) {
            print_uart("FAIL\r\n");
            errors++;
            
            // Dump first few mismatches
            for(int i=0; i<n && i<4; ++i) {
                print_uart("  i="); print_uart_int_dec(i);
                print_uart(" ref="); print_uart_int_dec((int)(ref_buf[i]*1000));
                print_uart(" rvv="); print_uart_int_dec((int)(rvv_buf[i]*1000));
                print_uart("\r\n");
            }

        } else {
            print_uart("PASS\r\n");
        }
    }

    if (errors == 0) {
        print_uart("ALL TESTS PASSED\r\n");
        exit(0);
    } else {
        print_uart("SOME TESTS FAILED\r\n");
        exit(1);
    }
    return 0;
}
