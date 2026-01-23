#include <riscv_vector.h>
#include "uart_helper.c"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "matrix_kernel.h"

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

static int float_equals_eps(float a, float b, float eps) {
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= eps;
}

static void print_result(const char *name, int ok) {
    print_uart(name);
    print_uart(": ");
    print_uart(ok ? "PASS\r\n" : "FAIL\r\n");
}

// helpers to print vectors
static void print_float_vector(const char *title, const float *v, int len) {
    /* Use uart_helper's print_f32_array to print indexed float values */
    print_f32_array(title, v, len);
}

static void print_i32_vector(const char *title, const int32_t *v, int len) {
    /* Print indexed int32 values using uart_helper primitives */
    print_uart(title);
    print_uart(" (len=");
    print_uart_int_dec(len);
    print_uart("):\r\n");
    for (int i = 0; i < len; ++i) {
        print_idx_prefix(title, i);
        print_dec32(v[i]);
        print_uart("\r\n");
    }
}

static void print_ws_strided(const char *title, const float *ws_base, ptrdiff_t ws_stride_bytes, int len) {
    /* Print each strided ws element using uart_helper primitives */
    print_uart(title);
    print_uart(" (len=");
    print_uart_int_dec(len);
    print_uart("):\r\n");
    for (int i = 0; i < len; ++i) {
        const float *p = (const float *)((const uint8_t *)ws_base + (size_t)i * (size_t)ws_stride_bytes);
        print_idx_prefix(title, i);
        print_float_fixed3(*p);
        print_uart("\r\n");
    }
}

static int test_accum_contiguous(void) {
    const int n = 128; // exercise non-multiple of typical vector lengths
    float dst[n];
    int32_t src[n];
    float ws[n];
    float xs = 0.5f;

    // init
    for (int i = 0; i < n; ++i) {
        dst[i] = (float)(i + 1);       // start with non-zero destination
        src[i] = (i % 7) - 3;          // some negative/positive values
        ws[i] = 0.1f * (float)(i + 1);
    }

    // print inputs
    print_uart("--- test_accum_contiguous inputs ---\r\n");
    print_float_vector("dst (before)", dst, n);
    print_i32_vector("src", src, n);
    print_float_vector("ws", ws, n);
    print_f32_scalar("xs", xs);

    // reference
    float ref[n];
    for (int i = 0; i < n; ++i) ref[i] = dst[i] + (float)src[i] * ws[i] * xs;
    print_float_vector("ref", ref, n);

    // call kernel: stride = sizeof(float)
    matrix_kernel_rvv_accum_i32_to_f32_ws_xs(dst, src, ws, (ptrdiff_t)sizeof(float), xs, n);

    // print outputs
    print_float_vector("dst (after)", dst, n);

    // compare and report mismatch details
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (!float_equals_eps(dst[i], ref[i], 1e-6f)) {
            ok = 0;
            print_idx_prefix("mismatch", i); print_float_fixed3(dst[i]); print_uart(" "); print_uart("exp="); print_float_fixed3(ref[i]); print_uart("\r\n");
            // continue to show more mismatches
        }
    }
    return ok;
} 

static int test_accum_strided(void) {
    // Strided path uses a fixed-size pack buffer in the kernel (MK_MMAX=32).
    // Keep n <= 32 to avoid overflow.
    const int n = 32;
    float dst[n];
    int32_t src[n];
    // allocate a buffer with stride 4 to simulate non-contiguous ws
    float ws_storage[4 * n];
    float *ws_base = &ws_storage[0];
    ptrdiff_t ws_stride_bytes = 4 * (ptrdiff_t)sizeof(float);
    float xs = -0.25f;

    // init
    for (int i = 0; i < n; ++i) {
        dst[i] = (float)(-i);
        src[i] = (i % 5) - 2;
        ws_storage[4 * i + 0] = 0.2f * (float)(i + 1);
        ws_storage[4 * i + 1] = 999.0f; // padding value should be ignored
        ws_storage[4 * i + 2] = 999.0f;
        ws_storage[4 * i + 3] = 999.0f;
    }

    // print inputs
    print_uart("--- test_accum_strided inputs ---\r\n");
    //print_float_vector("dst (before)", dst, n);
    print_i32_vector("src", src, n);
    print_ws_strided("ws (strided)", ws_base, ws_stride_bytes, n);
    print_f32_scalar("xs", xs);

    // reference
    float ref[n];
    for (int i = 0; i < n; ++i) {
        float ws_val = ws_storage[4 * i];
        ref[i] = dst[i] + (float)src[i] * ws_val * xs;
    }
    print_float_vector("ref", ref, n);

    matrix_kernel_rvv_accum_i32_to_f32_ws_xs(dst, src, ws_base, ws_stride_bytes, xs, n);

    // print outputs
    print_float_vector("dst (after)", dst, n);

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (!float_equals_eps(dst[i], ref[i], 1e-6f)) {
            ok = 0;
            print_idx_prefix("mismatch", i); print_float_fixed3(dst[i]); print_uart(" "); print_uart("exp="); print_float_fixed3(ref[i]); print_uart("\r\n");
        }
    }
    return ok;
}

// Cross-row fetch test: ws is laid out as ws[row][group] row-major.
// For a fixed group g, ws[row0 + i][g] are separated by row_stride=groups*sizeof(float).
static int test_accum_cross_row_ws(void) {
    // Keep n <= 32 for the kernel's strided-pack buffer.
    const int rows = 32;
    const int groups = 3; // groups_per_row > 1 ensures ws is truly strided across rows
    const int row0 = 5;
    const int g = 2;
    const int n = 16;

    float dst[n];
    int32_t src[n];
    float ws2d[rows * groups];
    ptrdiff_t ws_stride_bytes = (ptrdiff_t)groups * (ptrdiff_t)sizeof(float);
    float xs = 0.75f;

    // init ws matrix with unique values per (row, group)
    for (int r = 0; r < rows; ++r) {
        for (int gg = 0; gg < groups; ++gg) {
            ws2d[r * groups + gg] = 0.01f * (float)(r + 1) + 0.1f * (float)gg;
        }
    }
    for (int i = 0; i < n; ++i) {
        dst[i] = (float)(100 - i);
        src[i] = (i % 9) - 4;
    }

    const float *ws_base = &ws2d[row0 * groups + g];

    print_uart("--- test_accum_cross_row_ws inputs ---\r\n");
    print_float_vector("dst (before)", dst, n);
    print_i32_vector("src", src, n);
    print_uart("ws layout: ws[row][group] row-major\r\n");
    print_uart("row0="); print_uart_int_dec(row0);
    print_uart(" groups="); print_uart_int_dec(groups);
    print_uart(" g="); print_uart_int_dec(g);
    print_uart(" stride_bytes="); print_uart_int_dec((int)ws_stride_bytes);
    print_uart("\r\n");
    print_ws_strided("ws_column (cross-row)", ws_base, ws_stride_bytes, n);
    print_f32_scalar("xs", xs);

    float ref[n];
    for (int i = 0; i < n; ++i) {
        float ws_val = ws2d[(row0 + i) * groups + g];
        ref[i] = dst[i] + (float)src[i] * ws_val * xs;
    }
    print_float_vector("ref", ref, n);

    matrix_kernel_rvv_accum_i32_to_f32_ws_xs(dst, src, ws_base, ws_stride_bytes, xs, n);
    print_float_vector("dst (after)", dst, n);

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (!float_equals_eps(dst[i], ref[i], 1e-6f)) {
            ok = 0;
            print_idx_prefix("mismatch", i);
            print_float_fixed3(dst[i]);
            print_uart(" exp=");
            print_float_fixed3(ref[i]);
            print_uart("\r\n");
        }
    }
    return ok;
}

int real_main(void);

__attribute__((naked)) void main(void) {
    asm volatile(
        "li t0, 0x6600\n"      // VS=11, FS=11
        "csrs mstatus, t0\n"
        "j real_main\n"
    );
}

int real_main(void) {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("rvv_accum_test start\r\n\r\n");

    // int ok1 = test_accum_contiguous();
    // print_result("test_accum_contiguous", ok1);

    // int ok2 = test_accum_strided();
    // print_result("test_accum_strided", ok2);

    int ok3 = test_accum_cross_row_ws();
    print_result("test_accum_cross_row_ws", ok3);

    //int all_ok = ok1 && ok2;
    // print_uart("\r\n");
    // print_uart(all_ok ? "ALL TESTS PASS\r\n" : "SOME TESTS FAIL\r\n");

    while (1) { }
}
