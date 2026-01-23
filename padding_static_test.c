#include <riscv_vector.h>
#include "uart_helper.c"
#include <stdint.h>
#include <stddef.h>

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

// ---------------------------------------------------------------------------
// Force RVV_padding.c to use the static-buffer backend.
// Also make the buffers intentionally small so we can test overflow behavior.
#define RVV_PADDING_STATIC 1
#define RVV_PADDING_STATIC_1D_BYTES 128
#define RVV_PADDING_STATIC_2D_BYTES 256

// Ensure RVV_padding.c doesn't try to provide its own allocator.
#define RVV_PADDING_NO_ALLOCATOR 1

// If anything accidentally calls malloc/calloc in this test, trap.
void *malloc(size_t size) {
    (void)size;
    print_uart("ERROR: malloc() called in static padding test\r\n");
    while (1) { }
}

void *calloc(size_t count, size_t size) {
    (void)count;
    (void)size;
    print_uart("ERROR: calloc() called in static padding test\r\n");
    while (1) { }
}

void free(void *ptr) { (void)ptr; }

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

static int test_static_pad_array_basic(void) {
    float x[5] = {1, 2, 3, 4, 5};

    PaddedArray1D px = pad_array_1d(x, 5, sizeof(float), 8);
    if (!px.data) return 0;

    float *p = (float *)px.data;

    int ok = 1;
    for (int i = 0; i < 5; i++) {
        if (!float_equals_eps(p[i], x[i], 1e-6f)) ok = 0;
    }
    for (size_t i = 5; i < 8; i++) {
        if (!float_equals_eps(p[i], 0.0f, 1e-6f)) ok = 0;
    }

    free_padded_array_1d(&px);
    return ok;
}

static int test_static_pad_array_reentrancy_and_reuse(void) {
    float x[4] = {10, 20, 30, 40};

    PaddedArray1D a = pad_array_1d(x, 4, sizeof(float), 8);
    if (!a.data) return 0;

    // Second allocation without free should fail (single static buffer).
    PaddedArray1D b = pad_array_1d(x, 4, sizeof(float), 8);
    int ok = (b.data == NULL);

    void *ptr_a = a.data;
    free_padded_array_1d(&a);

    // After free, allocation should succeed again, typically reusing same buffer.
    PaddedArray1D c = pad_array_1d(x, 4, sizeof(float), 8);
    if (!c.data) ok = 0;
    if (c.data != ptr_a) ok = 0;

    free_padded_array_1d(&c);
    return ok;
}

static int test_static_pad_array_overflow_returns_null(void) {
    float x[1] = {1};

    // 128 bytes buffer; ask for 33 floats => 132 bytes.
    PaddedArray1D px = pad_array_1d(x, 1, sizeof(float), 33);
    return (px.data == NULL);
}

static int test_static_pad_matrix_basic(void) {
    float w[2][3] = {
        {1, 2, 3},
        {4, 5, 6},
    };

    PaddedArray2D pw = pad_matrix_2d(&w[0][0], 2, 3, sizeof(float), 4);
    if (!pw.data) return 0;

    float *p = (float *)pw.data;

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

static int test_static_pad_matrix_null_source_zero_init(void) {
    // 2 x 8 floats = 64 bytes, well under RVV_PADDING_STATIC_2D_BYTES.
    PaddedArray2D pm = pad_matrix_2d(NULL, 2, 5, sizeof(float), 8);
    if (!pm.data) return 0;

    float *p = (float *)pm.data;
    int ok = 1;
    for (size_t i = 0; i < 2 * 8; i++) {
        if (!float_equals_eps(p[i], 0.0f, 1e-6f)) ok = 0;
    }

    free_padded_matrix_2d(&pm);
    return ok;
}

static int test_static_pad_matrix_reentrancy_and_reuse(void) {
    float w[1][2] = {{7, 8}};

    PaddedArray2D a = pad_matrix_2d(&w[0][0], 1, 2, sizeof(float), 4);
    if (!a.data) return 0;

    PaddedArray2D b = pad_matrix_2d(&w[0][0], 1, 2, sizeof(float), 4);
    int ok = (b.data == NULL);

    void *ptr_a = a.data;
    free_padded_matrix_2d(&a);

    PaddedArray2D c = pad_matrix_2d(&w[0][0], 1, 2, sizeof(float), 4);
    if (!c.data) ok = 0;
    if (c.data != ptr_a) ok = 0;

    free_padded_matrix_2d(&c);
    return ok;
}

static int test_static_pad_matrix_overflow_returns_null(void) {
    // 256 bytes buffer; ask for 17x4 floats => 272 bytes.
    float dummy[1] = {0};
    PaddedArray2D pm = pad_matrix_2d(dummy, 17, 1, sizeof(float), 4);
    return (pm.data == NULL);
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
    print_uart("padding_static_test start\r\n\r\n");

    int ok1 = test_static_pad_array_basic();
    print_result("test_static_pad_array_basic", ok1);

    int ok2 = test_static_pad_array_reentrancy_and_reuse();
    print_result("test_static_pad_array_reentrancy_and_reuse", ok2);

    int ok3 = test_static_pad_array_overflow_returns_null();
    print_result("test_static_pad_array_overflow_returns_null", ok3);

    int ok4 = test_static_pad_matrix_basic();
    print_result("test_static_pad_matrix_basic", ok4);

    int ok5 = test_static_pad_matrix_null_source_zero_init();
    print_result("test_static_pad_matrix_null_source_zero_init", ok5);

    int ok6 = test_static_pad_matrix_reentrancy_and_reuse();
    print_result("test_static_pad_matrix_reentrancy_and_reuse", ok6);

    int ok7 = test_static_pad_matrix_overflow_returns_null();
    print_result("test_static_pad_matrix_overflow_returns_null", ok7);

    int all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7;
    print_uart("\r\n");
    print_uart(all_ok ? "ALL TESTS PASS\r\n" : "SOME TESTS FAIL\r\n");

    while (1) { }
}
