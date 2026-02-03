// Test harness for matrix_kernel_qmatmul_f32_noblk_batch
// Can be built using:
// make rvbareclang RV_BARE_APP=./260105_chat/test_matrix_kernel_batch.c BARE_PLATFORM=fpga PRINT_WAY=uart

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include your matrix kernel headers
#include "matrix_kernel_1230.h"
#include "matrix_kernel_noblk_1231.h"
#include "uart_helper.c" // For bare-metal print

// Define BARE_NO_AUTOVEC if not included from headers
#ifndef BARE_NO_AUTOVEC
#if defined(__clang__)
#define BARE_NO_AUTOVEC __attribute__((optnone, noinline))
#else
#define BARE_NO_AUTOVEC
#endif
#endif

// External declaration if not in header
void matrix_kernel_qmatmul_f32_noblk_batch(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride,
                                      int batch);

// Simple RNG for deterministic tests
static uint64_t rng_state = 123456789;
static float rand_float(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1;
    float r = (float)(rng_state >> 33) / 2147483648.0f; // [0, 1)
    return r * 2.0f - 1.0f; // [-1, 1]
}

static int8_t rand_i8(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1;
    return (int8_t)(rng_state >> 56);
}

// Reference implementation (Scalar)
static void matmul_ref(float *out, const int8_t *xq, const float *xs, 
                       const int8_t *wq, const float *ws, 
                       int n, int d, int gs, int batch) {
    int num_groups = (n + gs - 1) / gs;

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < d; i++) { // Output dimension (rows of W, cols of Output if we view as Batch x D)
            float acc = 0.0f;
            for (int j = 0; j < n; j++) {
                int group = j / gs;
                // W is (d, n)
                int8_t w_val = wq[i * n + j];
                float w_scale = ws[i * num_groups + group];
                
                // X is (batch, n)
                int8_t x_val = xq[b * n + j];
                float x_scale = xs[b * num_groups + group]; // Corrected index for xs
                
                acc += ((float)w_val * w_scale) * ((float)x_val * x_scale);
            }
            // Output is (batch, d)
            out[b * d + i] = acc;
        }
    }
}

// Minimal fabsf replacement for bare-metal builds (avoid linking libm)
static inline float my_fabsf(float x) {
    return x < 0.0f ? -x : x;
}

static inline int8_t q_round_clamp_i8(float x) {
    int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    return (int8_t)qi;
}

static void quantize_per_group_i8(const float *x, int n, int gs, int8_t *xq, float *xs) {
    const int num_groups = (n + gs - 1) / gs;
    const float Q_MAX = 127.0f;

    for (int g = 0; g < num_groups; g++) {
        const int base = g * gs;
        const int count = (base + gs <= n) ? gs : (n - base);

        float xmax = 0.0f;
        for (int i = 0; i < count; i++) {
            float v = my_fabsf(x[base + i]);
            if (v > xmax) xmax = v;
        }

        float scale = (xmax > 0.0f) ? (xmax / Q_MAX) : 1.0f;
        xs[g] = scale;

        if (xmax == 0.0f) {
            for (int i = 0; i < count; i++) xq[base + i] = 0;
        } else {
            float inv = 1.0f / scale;
            for (int i = 0; i < count; i++) {
                xq[base + i] = q_round_clamp_i8(x[base + i] * inv);
            }
        }
    }
}


// Minimal UART and printing stubs for bare-metal test builds.
// These satisfy linker symbols expected by the build so the test can run
// in a host environment or simple bare-metal harness.
/*
void write_serial(char a) {
    (void)a;
}

void init_uart(uint32_t freq, uint32_t baud) {
    (void)freq; (void)baud;
}

void print_uart(const char *s) {
    while (*s) write_serial(*s++);
}
*/

// Provide a simple wrapper implementation of the hardware kernel symbol
// so the test can link and run in a reference mode. It forwards to
// the scalar reference implementation above.
/*
void matrix_kernel_qmatmul_f32_noblk_batch(float *xout,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq,
                                           const float *ws,
                                           int n,
                                           int d,
                                           int gs,
                                           int w_row_stride,
                                           int batch) {
    (void)w_row_stride;
    matmul_ref(xout, xq, xs, wq, ws, n, d, gs, batch);
}
*/

// Compare results
static int check_result(const float *ref, const float *out, int size, float tol) {
    float max_diff = 0.0f;
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = my_fabsf(ref[i] - out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (errors < 5) {
                // Manually printing for bare metal env
                // print_uart("Error at index "); print_uart_int(i); 
                // ... (simplified)
            }
            errors++;
        }
    }
    init_uart(50000000, 115200);
    print_uart("Max Error: "); 
    // basic print float implementation might be needed or just cast to int for simple check
    int diff_int = (int)(max_diff * 10000);
    print_uart_int_dec(diff_int); 
    print_uart(" * 1e-4\r\n");

    return errors;
}

// Enable RVV (Copy from run.c)
static inline void enable_rvv_state(void) {
    uintptr_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9); // VS[10:9] = 0b11
    mstatus |= (3UL << 13);  // FS=Dirty
    mstatus |= (3UL << 15);  // XS[16:15] = Dirty
    __asm__ volatile ("csrw mstatus, %0" :: "r"(mstatus) : "memory");
}

#define N 64
#define D 32
#define BATCH 4
#define GS 32

// Buffers (Static to avoid stack overflow in bare metal)
static float   X_f32[BATCH * N];
static float   W_f32[D * N];
static int8_t  Xq[BATCH * N];
static float   Xs[BATCH * (N/GS)];
static int8_t  Wq[D * N];
static float   Ws[D * (N/GS)];
static float   Out_HW[BATCH * D];
static float   Out_Ref[BATCH * D];

int main(void) {
    enable_rvv_state();
    init_uart(50000000, 115200);
    print_uart("\r\n=== Test Matrix Kernel Batch ===\r\n");

    int n = N;
    int d = D;
    int batch = BATCH;
    int gs = GS;
    int num_groups = (n + gs - 1) / gs;


    // Initialize data
    print_uart("Initializing random data...\r\n");
    // Generate float data
    for (int i = 0; i < BATCH * N; i++) X_f32[i] = rand_float() * 0.7f;
    for (int i = 0; i < D * N; i++) W_f32[i] = rand_float() * 0.7f;

    // Quantize X (Batch x N) -> Xq, Xs
    quantize_per_group_i8(X_f32, BATCH * N, GS, Xq, Xs);
    for (int b = 0; b < batch; b++) {
        print_uart("Input Quantized X (Batch ");
        print_uart_int_dec((uint64_t)b);
        print_uart("):\r\n");
        for (int i = 0; i < n; i++) {
            print_dec32((int32_t)Xq[b * n + i]);
            print_uart((i % 8 == 7) ? "\r\n" : " ");
        }
        print_uart("\r\n");

        print_uart("Input Scales Xs (Batch ");
        print_uart_int_dec((uint64_t)b);
        print_uart("):\r\n");
        for (int i = 0; i < num_groups; i++) {
            print_float_fixed3(Xs[b * num_groups + i]);
            print_uart(" ");
        }
        print_uart("\r\n");
    }

    // Quantize W (D x N) -> Wq, Ws
    quantize_per_group_i8(W_f32, D * N, GS, Wq, Ws); 

    print_uart("Weight Quantized W (Row 0):\r\n");
    for (int i = 0; i < n; i++) {
        print_dec32((int32_t)Wq[i]);
        print_uart((i % 8 == 7) ? "\r\n" : " ");
    }
    print_uart("\r\n");

    print_uart("Weight Scales Ws (Row 0):\r\n");
    for (int i = 0; i < num_groups; i++) {
        print_float_fixed3(Ws[i]);
        print_uart(" ");
    }
    print_uart("\r\n");
    // Compute Reference
    print_uart("Computing Reference...\r\n");
    matmul_ref(Out_Ref, Xq, Xs, Wq, Ws, n, d, gs, batch);

    for (int b = 0; b < batch; b++) {
        print_uart("Ref Output (Batch ");
        print_uart_int_dec((uint64_t)b);
        print_uart("):\r\n");
        for (int i = 0; i < d; i++) {
            print_float_fixed3(Out_Ref[b * d + i]);
            print_uart((i % 8 == 7) ? "\r\n" : " ");
        }
        print_uart("\r\n");
    }

    // Compute HW
    print_uart("Computing HW Batch Kernel...\r\n");
    // Clear output first (kernel expects it or zeros it? our kernel zeros it)
    //memset(Out_HW, 0, sizeof(Out_HW)); 
    
    matrix_kernel_qmatmul_f32_noblk_batch(Out_HW, Xq, Xs, Wq, Ws, n, d, gs, n, batch);

    for (int b = 0; b < batch; b++) {
        print_uart("HW Output (Batch ");
        print_uart_int_dec((uint64_t)b);
        print_uart("):\r\n");
        for (int i = 0; i < d; i++) {
            print_float_fixed3(Out_HW[b * d + i]);
            print_uart((i % 8 == 7) ? "\r\n" : " ");
        }
        print_uart("\r\n");
    }

    // Compare
    print_uart("Verifying...\r\n");
    int errors = check_result(Out_Ref, Out_HW, BATCH * D, 0.1f); // 0.1 tolerance is loose but quantization/float order differs

    if (errors == 0) {
        print_uart("[PASS] Results match.\r\n");
    } else {
        print_uart("[FAIL] Mismatch found.\r\n");
        print_uart("Total failures: "); print_uart_int_dec(errors); print_uart("\r\n");
    }

    // Benchmark (Optional)
    // ...

    return 0;
}
