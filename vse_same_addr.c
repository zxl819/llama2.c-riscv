#include <riscv_vector.h>
#include "uart_helper.c"
#include <string.h>
#include <stdint.h>

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) asm volatile("nop");
}

static void __attribute__((noinline)) print_floats(const char *tag, const float *buf, size_t count) {
    print_uart(tag);
    for (size_t i = 0; i < count; i++) {
        print_uart_float(buf[i]);
        print_uart(" ");
        if (((i + 1) % 8) == 0) print_uart("\r\n");
    }
    if ((count % 8) != 0) print_uart("\r\n");
}

static void __attribute__((noinline)) run_case(const char *name, float *dst, const float *src1, const float *src2, size_t buf_len, size_t offset_floats) {
    print_uart("\r\n== ");
    print_uart(name);
    print_uart(" ==\r\n");

    float *ptr = dst + offset_floats;
    size_t remaining = buf_len - offset_floats;
    size_t vl = __riscv_vsetvl_e32m1(remaining > 16 ? 16 : remaining);

    print_uart("store addr=0x");
    print_uart_hex((unsigned long long)(uintptr_t)ptr);
    print_uart(" offset_floats=");
    print_uart_int_dec((int)offset_floats);
    print_uart(" vl=");
    print_uart_int_dec((int)vl);
    print_uart("\r\n");

    vfloat32m1_t v1 = __riscv_vle32_v_f32m1(src1 + offset_floats, vl);
    __riscv_vse32_v_f32m1(ptr, v1, vl);
    debug_delay_cycles(100);

    size_t window = vl + 4 <= remaining ? vl + 4 : remaining;
    print_uart("after src1->dst:\r\n");
    print_floats("window:\r\n", ptr, window);

    int mismatches = 0;
    for (size_t i = 0; i < vl; i++) {
        float expected = src1[offset_floats + i];
        float got = ptr[i];
        if (got != expected) {
            mismatches++;
            if (mismatches <= 8) {
                print_uart("src1 mismatch i=");
                print_uart_int_dec((int)i);
                print_uart(" got=");
                print_uart_float(got);
                print_uart(" expected=");
                print_uart_float(expected);
                print_uart("\r\n");
            }
        }
    }
    if (mismatches == 0) {
        print_uart("src1 result: ALL OK\r\n");
    } else {
        print_uart("src1 result: mismatches=");
        print_uart_int_dec(mismatches);
        print_uart("\r\n");
    }

    vfloat32m1_t v2 = __riscv_vle32_v_f32m1(src2 + offset_floats, vl);
    __riscv_vse32_v_f32m1(ptr, v2, vl);
    debug_delay_cycles(100);

    print_uart("after src2->dst:\r\n");
    print_floats("window:\r\n", ptr, window);

    mismatches = 0;
    for (size_t i = 0; i < vl; i++) {
        float expected = src2[offset_floats + i];
        float got = ptr[i];
        if (got != expected) {
            mismatches++;
            if (mismatches <= 8) {
                print_uart("src2 mismatch i=");
                print_uart_int_dec((int)i);
                print_uart(" got=");
                print_uart_float(got);
                print_uart(" expected=");
                print_uart_float(expected);
                print_uart("\r\n");
            }
        }
    }
    if (mismatches == 0) {
        print_uart("src2 result: ALL OK\r\n");
    } else {
        print_uart("src2 result: mismatches=");
        print_uart_int_dec(mismatches);
        print_uart("\r\n");
    }
}

int real_main();

__attribute__((naked)) void main() {
    asm volatile(
        "li t0, 0x6600\n"      // VS=11, FS=11
        "csrs mstatus, t0\n"
        "j real_main\n");
}

int real_main() {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("\r\n[vse_same_addr] start\r\n");

    size_t vlmax = __riscv_vsetvlmax_e32m1();
    (void)vlmax;

    __attribute__((aligned(64))) float dst[64];
    __attribute__((aligned(64))) float src1[64];
    __attribute__((aligned(64))) float src2[64];

    for (size_t i = 0; i < 64; i++) {
        dst[i] = -1.0f;
        src1[i] = 10.0f + (float)(i + 1);
        src2[i] = 20.0f + (float)(i + 1);
    }

    // Two distinct sources stored into the same destination to check hazards at different offsets.
    run_case("store to dst+0", dst, src1, src2, 32, 0);

    print_uart("\r\n[vse_same_addr] done\r\n");
    while (1) asm volatile("wfi");
    return 0;
}
