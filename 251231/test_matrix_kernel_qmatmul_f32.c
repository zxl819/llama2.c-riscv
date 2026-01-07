// Minimal bare-metal correctness test for matrix_kernel_qmatmul_f32
//
// Build (example):
//   make rvbareclang RV_BARE_APP=./test_matrix_kernel_qmatmul_f32.c \
//     RV_BARE_NAME=./test_matrix_kernel_qmatmul_f32 BARE_PLATFORM=fpga PRINT_WAY=uart \
//     BARE_EMBED_BLOBS=0 RV_BARE_CLANG=~/newllvm/test_llvm/build/bin/clang \
//     CFLAGS+='-O1 -DMATRIX_KERNEL_DEBUG_PRINT=0 -DMATRIX_KERNEL_DEBUG_PRINT_EVERY=0'
//
// Notes:
// - This test assumes the platform supports the Matrix extension used by matrix_kernel.h.
// - It avoids libm; only uses basic float ops.

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "uart_helper.c"

// 显式禁用内核调试打印，消除 debug_delay_cycles 和额外的栈开销，使汇编更清晰

#include "matrix_kernel_1230.h"
#include "matrix_kernel_noblk_1231.h"

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

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

static inline void enable_vector_state(void) {
    uintptr_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9);   // VS=Dirty
    mstatus |= (uintptr_t)(3u << 13);  // FS=Dirty
    mstatus |= (uintptr_t)(3u << 15);  // XS=Dirty
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
// Small deterministic PRNG (xorshift64*)
/*rng_u32()：生成一个 32 位伪随机整数（PRNG）

用的是 xorshift64* 算法：对全局状态 g_rng 做几次异或移位（x ^= x >> 12 / x ^= x << 25 / x ^= x >> 27），再乘一个常数做“搅拌”，最后取高 32 位返回。
每调用一次都会更新 g_rng，所以会得到一串确定的随机序列（种子固定时完全可复现）。
rng_f32_signed()：把 rng_u32() 转成一个浮点数，范围大致在 [-1, 1]

u >> 8 取高 24 位（这样能刚好映射到 float 的“有效随机精度”附近）
*(1.0f / 16777216.0f) 把它缩放到 [0, 1)
最后 f * 2 - 1 映射到 [-1, 1]
*/
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
    // Uniform-ish in [-1, 1]
    uint32_t u = rng_u32();
    float f = (float)(u >> 8) * (1.0f / 16777216.0f); // [0,1)
    return (f * 2.0f) - 1.0f;
}

static inline float q_fabsf(float x) { return x < 0.0f ? -x : x; }

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
            float v = q_fabsf(x[base + i]);
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

static void ref_qmatmul_f32(float *out,
                           const int8_t *xq, const float *xs,
                           const int8_t *wq, const float *ws,
                           int n, int d, int gs) {
    const int num_groups = (n + gs - 1) / gs;
    const int groups_per_row = n / gs;

    for (int i = 0; i < d; i++) {
        float accf = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            const int base = g * gs;
            const int count = (base + gs <= n) ? gs : (n - base);

            int32_t acc = 0;
            const int8_t *wrow = wq + (ptrdiff_t)i * (ptrdiff_t)n + (ptrdiff_t)base;
            const int8_t *xvec = xq + (ptrdiff_t)base;
            for (int j = 0; j < count; j++) {
                acc += (int32_t)wrow[j] * (int32_t)xvec[j];
            }

            const float wscale = ws[(ptrdiff_t)i * (ptrdiff_t)groups_per_row + (ptrdiff_t)g];
            const float xscale = xs[g];
            accf += ((float)acc) * wscale * xscale;
        }
        out[i] = accf;
    }
}

static int check_close(const float *a, const float *b, int n, float atol, float rtol) {
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        if (diff < 0.0f) diff = -diff;
        float thresh = atol + rtol * (q_fabsf(b[i]));
        if (!(diff <= thresh)) return i + 1; // 1-based index for reporting
    }
    return 0;
}

static void print_i32_dec(int32_t v) {
    if (v < 0) {
        write_serial((uint8_t)'-');
        // handle INT32_MIN safely
        uint32_t uv = (uint32_t)(-(v + 1)) + 1u;
        print_uart_int_dec((uint64_t)uv);
        return;
    }
    print_uart_int_dec((uint64_t)v);
}

static void print_vec_f32(const char *label, const float *v, int n) {
    print_uart(label);
    print_uart("\r\n");
    for (int i = 0; i < n; i++) {
        print_uart("  [");
        print_uart_int_dec((uint64_t)i);
        print_uart("]=");
        print_float_fixed3(v[i]);
        print_uart("\r\n");
    }
}

static void print_vec_i8(const char *label, const int8_t *v, int n) {
    print_uart(label);
    print_uart("\r\n");
    for (int i = 0; i < n; i++) {
        print_uart("  [");
        print_uart_int_dec((uint64_t)i);
        print_uart("]=");
        print_i32_dec((int32_t)v[i]);
        print_uart("\r\n");
    }
}

static void print_mat_f32(const char *label, const float *m, int rows, int cols) {
    print_uart(label);
    print_uart("\r\n");
    for (int r = 0; r < rows; r++) {
        print_uart("  r=");
        print_uart_int_dec((uint64_t)r);
        print_uart(": ");
        for (int c = 0; c < cols; c++) {
            if (c) write_serial((uint8_t)' ');
            print_float_fixed3(m[r * cols + c]);
        }
        print_uart("\r\n");
    }
}

static void print_mat_i8(const char *label, const int8_t *m, int rows, int cols) {
    print_uart(label);
    print_uart("\r\n");
    for (int r = 0; r < rows; r++) {
        print_uart("  r=");
        print_uart_int_dec((uint64_t)r);
        print_uart(": ");
        for (int c = 0; c < cols; c++) {
            if (c) write_serial((uint8_t)' ');
            print_i32_dec((int32_t)m[r * cols + c]);
        }
        print_uart("\r\n");
    }
}

int main(void) {
    enable_vector_state();
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    print_uart("[test] matrix_kernel_qmatmul_f32...\r\n");

    // Test sizes.
    // IMPORTANT: matrix_kernel_qmatmul_f32 assumes N is a multiple of GS so that
    // ws indexing becomes affine per row (groups_per_row = N/GS).
    // It also assumes per-group K (count) <= 32 (MK_KMAX).
    enum { GS = 32 };
    enum { N = 32 };  // input dimension (model dim)
    enum { D = 64 };  // output dimension (hidden_dim)

    _Static_assert(GS > 0, "GS must be > 0");
    _Static_assert(GS <= 32, "This test expects GS<=32 (kernel MK_KMAX=32)");
    _Static_assert((N % GS) == 0, "This test expects N%GS==0; otherwise ws indexing in matrix_kernel_qmatmul_f32 is not valid");

    static float x[N] __attribute__((aligned(64)));
    static float w[D * N] __attribute__((aligned(64)));

    static int8_t xq[N] __attribute__((aligned(64)));
    static float xs[(N + GS - 1) / GS] __attribute__((aligned(64)));

    static int8_t wq[D * N] __attribute__((aligned(64)));
    static float ws[(D * N + GS - 1) / GS] __attribute__((aligned(64)));

    static float out[D] __attribute__((aligned(64)));
    //__attribute__((section(".matNOLOAD"), aligned(64))) static
    static float ref[D] __attribute__((aligned(64)));

    // Fill float inputs with small-ish values to reduce saturation.
    for (int i = 0; i < N; i++) x[i] = rng_f32_signed() * 0.7f;
    for (int i = 0; i < D * N; i++) w[i] = rng_f32_signed() * 0.7f;

    // // Print float inputs
    // print_vec_f32("[test] x (f32)", x, N);
    // print_mat_f32("[test] w (f32) row-major [D x N]", w, D, N);

    // Quantize x per group
    quantize_per_group_i8(x, N, GS, xq, xs);

    // Quantize w over flattened array (matches kernel assumption when N%GS==0)
    quantize_per_group_i8(w, D * N, GS, wq, ws);

    // // Print quantized inputs
    // print_vec_i8("[test] xq (i8)", xq, N);
    // print_vec_f32("[test] xs (scale)", xs, (N + GS - 1) / GS);
    // print_mat_i8("[test] wq (i8) row-major [D x N]", wq, D, N);
    // print_vec_f32("[test] ws (scale over flattened w)", ws, (D * N + GS - 1) / GS);

    // Run kernel (use noblk variant) and reference
    matrix_kernel_qmatmul_f32_noblk(out, xq, xs, wq, ws, N, D, GS, N);
    print_uart_hex((uint64_t)(uintptr_t)out);
    debug_delay_cycles(100);
    ref_qmatmul_f32(ref, xq, xs, wq, ws, N, D, GS);

    // Always print results for debugging (even on PASS).
    print_vec_f32("[test] out", out, D);
    print_vec_f32("[test] ref", ref, D);

    // Tolerances: exact match is expected (same arithmetic order) unless
    // the kernel uses different rounding/accumulation. Keep a tiny epsilon.
    const float ATOL = 1e-4f;
    const float RTOL = 1e-4f;
    int bad = check_close(out, ref, D, ATOL, RTOL);

    if (bad) {
        print_uart("[test] FAIL at i=");
        print_uart_int_dec((uint64_t)(bad - 1));
        print_uart(" out=");
        print_float_fixed3(out[bad - 1]);
        print_uart(" ref=");
        print_float_fixed3(ref[bad - 1]);
        print_uart("\r\n");
        //exit(1);
    }

    print_uart("[test] PASS\r\n");
    return 0;
}
