#include <riscv_vector.h>
#include "uart_helper.c"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

// ---------------------------------------------------------------------------
// Tiny bump allocator for bare-metal use
#ifndef BARE_HEAP_BYTES
#define BARE_HEAP_BYTES (4 * 1024 * 1024)
#endif
static unsigned char bare_heap[BARE_HEAP_BYTES];
static size_t bare_heap_offset = 0;

static void *bare_alloc(size_t size, size_t alignment) {
    if (alignment == 0) alignment = 8;
    size_t misalignment = bare_heap_offset % alignment;
    if (misalignment != 0) bare_heap_offset += alignment - misalignment;
    if (bare_heap_offset + size > BARE_HEAP_BYTES) {
        print_uart("[bare] heap exhausted\r\n");
        while (1) { }
    }
    void *ptr = &bare_heap[bare_heap_offset];
    bare_heap_offset += size;
    return ptr;
}

void *malloc(size_t size) {
    if (size == 0) size = 1;
    return bare_alloc(size, 8);
}

void *calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return bare_alloc(1, 8);
    size_t total = count * size;
    void *ptr = bare_alloc(total, 8);
    memset(ptr, 0, total);
    return ptr;
}

void free(void *ptr) { (void)ptr; }

#define RVV_PADDING_NO_ALLOCATOR 1
#include "RVV_padding.c"

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

static void print_float_vector(const char *title, const float *v, size_t len) {
    print_uart(title);
    print_uart(" (len=");
    print_uart_int_dec((int)len);
    print_uart("):\r\n");
    for (size_t i = 0; i < len; i++) {
        print_uart_float(v[i]);
        print_uart(" ");
        if (((int)i + 1) % 8 == 0) print_uart("\r\n");
    }
    print_uart("\r\n");
}

static void print_float_matrix(const char *title, const float *m, size_t rows, size_t cols) {
    print_uart(title);
    print_uart(" (rows=");
    print_uart_int_dec((int)rows);
    print_uart(" cols=");
    print_uart_int_dec((int)cols);
    print_uart("):\r\n");
    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        print_uart_float(m[i]);
        print_uart(" ");
        if (((int)i + 1) % 8 == 0) print_uart("\r\n");
    }
    if ((total % 8) != 0) print_uart("\r\n");
    print_uart("\r\n");
}

static void print_i32_matrix(const char *title, const int32_t *m, size_t rows, size_t cols) {
    print_uart(title);
    print_uart(" (rows=");
    print_uart_int_dec((int)rows);
    print_uart(" cols=");
    print_uart_int_dec((int)cols);
    print_uart("):\r\n");
    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        print_uart_int_dec(m[i]);
        print_uart(" ");
        if (((int)i + 1) % 8 == 0) print_uart("\r\n");
    }
    if ((total % 8) != 0) print_uart("\r\n");
    print_uart("\r\n");
}

static int test_pad_array_1d_print(void) {
    float x[5] = {1, 2, 3, 4, 5};
    size_t padded_len = 8;

    print_float_vector("orig x", x, 5);

    PaddedArray1D px = pad_array_1d(x, 5, sizeof(float), padded_len);
    if (!px.data) return 0;

    float *p = (float *)px.data;
    print_float_vector("padded x", p, padded_len);

    int ok = 1;
    for (int i = 0; i < 5; i++) {
        if (!float_equals_eps(p[i], x[i], 1e-6f)) ok = 0;
    }
    for (size_t i = 5; i < padded_len; i++) {
        if (!float_equals_eps(p[i], 0.0f, 1e-6f)) ok = 0;
    }

    free_padded_array_1d(&px);
    return ok;
}

static int test_pad_matrix_2x3_to_2x4_print(void) {
    float w[2][3] = {
        {1, 2, 3},
        {4, 5, 6},
    };

    print_float_matrix("orig w", &w[0][0], 2, 3);

    PaddedArray2D pw = pad_matrix_2d(&w[0][0], 2, 3, sizeof(float), 4);
    if (!pw.data) return 0;

    float *p = (float *)pw.data;
    print_float_matrix("padded w", p, pw.rows, pw.padded_cols);

    int ok = 1;
    if (!float_equals_eps(p[0], 1.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[1], 2.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[2], 3.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[3], 0.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[4], 4.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[5], 5.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[6], 6.0f, 1e-6f)) ok = 0;
    if (!float_equals_eps(p[7], 0.0f, 1e-6f)) ok = 0;

    free_padded_matrix_2d(&pw);
    return ok;
}

static int test_pad_matrix_null_source_print(void) {
    size_t rows = 3;
    size_t original_cols = 5;
    size_t padded_cols = 8;

    PaddedArray2D pm = pad_matrix_2d(NULL, rows, original_cols, sizeof(float), padded_cols);
    if (!pm.data) return 0;

    float *p = (float *)pm.data;
    print_float_matrix("padded(NULL) matrix", p, pm.rows, pm.padded_cols);

    int ok = 1;
    for (size_t i = 0; i < rows * padded_cols; i++) {
        if (!float_equals_eps(p[i], 0.0f, 1e-6f)) ok = 0;
    }

    free_padded_matrix_2d(&pm);
    return ok;
}

static int test_pad_matrix_i32_print(void) {
    int32_t w[2][2] = {
        {10, 20},
        {30, 40},
    };

    print_i32_matrix("orig i32 w", &w[0][0], 2, 2);

    PaddedArray2D pw = pad_matrix_2d(&w[0][0], 2, 2, sizeof(int32_t), 4);
    if (!pw.data) return 0;

    int32_t *p = (int32_t *)pw.data;
    print_i32_matrix("padded i32 w", p, pw.rows, pw.padded_cols);

    int ok = 1;
    if (p[0] != 10 || p[1] != 20 || p[2] != 0 || p[3] != 0) ok = 0;
    if (p[4] != 30 || p[5] != 40 || p[6] != 0 || p[7] != 0) ok = 0;

    free_padded_matrix_2d(&pw);
    return ok;
}

static int test_pad_matrix_15x15_to_16x16_print(void) {
    // pad_matrix_2d 只能 pad 列数，不会自动 pad 行数。
    // 为了得到 16x16，这里先构造一个 16x15 的源矩阵：前 15 行是原始 15x15，最后 1 行补 0。
    float src16x15[16][15];
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 15; c++) {
            if (r < 15) {
                // 让值可读、可区分：100*r + c
                src16x15[r][c] = (float)(100 * r + c);
            } else {
                src16x15[r][c] = 0.0f;
            }
        }
    }

    print_float_matrix("orig 16x15 (from 15x15)", &src16x15[0][0], 16, 15);

    PaddedArray2D pw = pad_matrix_2d(&src16x15[0][0], 16, 15, sizeof(float), 16);
    if (!pw.data) return 0;

    float *p = (float *)pw.data;
    print_float_matrix("padded 16x16", p, pw.rows, pw.padded_cols);

    int ok = 1;
    // 校验：前 15x15 区域保持一致
    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            float expected = (float)(100 * r + c);
            if (!float_equals_eps(p[(size_t)r * 16 + (size_t)c], expected, 1e-6f)) ok = 0;
        }
    }
    // 校验：第 16 列（索引 15）前 15 行为 0
    for (int r = 0; r < 15; r++) {
        if (!float_equals_eps(p[(size_t)r * 16 + 15], 0.0f, 1e-6f)) ok = 0;
    }
    // 校验：第 16 行（索引 15）全部为 0（因为我们源矩阵最后一行就是 0）
    for (int c = 0; c < 16; c++) {
        if (!float_equals_eps(p[(size_t)15 * 16 + (size_t)c], 0.0f, 1e-6f)) ok = 0;
    }

    free_padded_matrix_2d(&pw);
    return ok;
}

static int test_pad_matrix_1x15_to_1x16_print(void) {
    float row[15];
    for (int i = 0; i < 15; i++) row[i] = (float)(i + 1);

    print_float_matrix("orig 1x15", row, 1, 15);

    PaddedArray2D pw = pad_matrix_2d(row, 1, 15, sizeof(float), 16);
    if (!pw.data) return 0;

    float *p = (float *)pw.data;
    print_float_matrix("padded 1x16", p, pw.rows, pw.padded_cols);

    int ok = 1;
    for (int c = 0; c < 15; c++) {
        if (!float_equals_eps(p[c], row[c], 1e-6f)) ok = 0;
    }
    if (!float_equals_eps(p[15], 0.0f, 1e-6f)) ok = 0;

    free_padded_matrix_2d(&pw);
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
    print_uart("padding_test start\r\n\r\n");

    int ok1 = test_pad_array_1d_print();
    print_result("test_pad_array_1d_print", ok1);

    int ok2 = test_pad_matrix_2x3_to_2x4_print();
    print_result("test_pad_matrix_2x3_to_2x4_print", ok2);

    int ok3 = test_pad_matrix_null_source_print();
    print_result("test_pad_matrix_null_source_print", ok3);

    int ok4 = test_pad_matrix_i32_print();
    print_result("test_pad_matrix_i32_print", ok4);

    int ok5 = test_pad_matrix_15x15_to_16x16_print();
    print_result("test_pad_matrix_15x15_to_16x16_print", ok5);

    int ok6 = test_pad_matrix_1x15_to_1x16_print();
    print_result("test_pad_matrix_1x15_to_1x16_print", ok6);

    int all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
    print_uart("\r\n");
    print_uart(all_ok ? "ALL TESTS PASS\r\n" : "SOME TESTS FAIL\r\n");

    while (1) { }
}
