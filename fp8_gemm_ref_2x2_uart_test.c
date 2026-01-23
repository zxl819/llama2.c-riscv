#include <stdint.h>
#include <stddef.h>

#include "uart_helper.c"
#include "fp8_e4m3fn.h"

// UART settings (match other bare tests)
#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

static inline void debug_delay_cycles(unsigned cycles) {
#if defined(__riscv)
    for (unsigned i = 0; i < cycles; ++i) {
        __asm__ volatile("nop");
    }
#else
    (void)cycles;
#endif
}

static void print_f32(float x) {
    // Minimal float printer: prints sign and scaled integer (x*1000)
    // This keeps the test freestanding-friendly.
    if (x != x) {
        print_uart("nan");
        return;
    }
    if (x < 0.0f) {
        write_serial((uint8_t)'-');
        x = -x;
    }
    int32_t scaled = (int32_t)(x * 1000.0f + 0.5f);
    int32_t ip = scaled / 1000;
    int32_t fp = scaled % 1000;
    print_dec32((int32_t)ip);
    write_serial((uint8_t)'.');
    // zero-pad 3 digits
    write_serial((uint8_t)('0' + (uint8_t)((fp / 100) % 10)));
    write_serial((uint8_t)('0' + (uint8_t)((fp / 10) % 10)));
    write_serial((uint8_t)('0' + (uint8_t)(fp % 10)));
}

static void matmul_f32_ref(float *C, const float *A, const float *B, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

int main(void) {
    // Include the fp8 test vectors.
    #include "data_2x2_fp8_e4m3fn.h"

    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("[fp8] E4M3FN 2x2 GEMM test\r\n");

    float A_f32[M*K];
    float B_f32[K*N];
    float C_f32[M*N];

    fp8e4m3fn_mat_to_f32(A_f32, A_fp8, M, K);
    fp8e4m3fn_mat_to_f32(B_f32, B_fp8, K, N);
    matmul_f32_ref(C_f32, A_f32, B_f32, M, N, K);

    print_uart("C calc (scaled x1000):\r\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            print_f32(C_f32[i * N + j]);
            if (j + 1 < N) write_serial((uint8_t)' ');
        }
        print_uart("\r\n");
    }

    print_uart("C ref:\r\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            print_f32(C_ref[i * N + j]);
            if (j + 1 < N) write_serial((uint8_t)' ');
        }
        print_uart("\r\n");
    }

    int mism = 0;
    for (int i = 0; i < M * N; ++i) {
        float d = C_f32[i] - C_ref[i];
        if (d < 0.0f) d = -d;
        if (d > 0.001f) mism++;
    }

    if (mism == 0) {
        print_uart("Compare: PASS\r\n");
    } else {
        print_uart("Compare: FAIL mismatches=");
        print_dec32((int32_t)mism);
        print_uart("\r\n");
    }

    debug_delay_cycles(1000000);
    return 0;
}
