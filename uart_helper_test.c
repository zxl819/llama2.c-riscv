// Standalone bare-metal tests for functions in uart_helper.c
//
// Run on Spike (tohost printf backend):
//   make rvrunbare RV_BARE_APP=uart_helper_test.c BARE_EMBED_BLOBS=0 PRINT_WAY=printf
//
// Build for hardware UART backend:
//   make rvbare RV_BARE_APP=uart_helper_test.c BARE_EMBED_BLOBS=0 PRINT_WAY=uart

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

// Intentionally include as a .c so we can directly call its static helpers.
#include "uart_helper.c"

#ifndef CLOCK_FREQUENCY
#define CLOCK_FREQUENCY 50000000u
#endif

#ifndef UART_BITRATE
#define UART_BITRATE 115200u
#endif

extern void _exit(int code);

// Override the weak trap handler from crt.S so failures are visible.
void handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t sp) {
    (void)sp;
    print_uart("\r\n*** TRAP ***\r\nmcause=");
    print_uart_hex((unsigned long long)mcause);
    print_uart(" mepc=");
    print_uart_hex((unsigned long long)mepc);
    print_uart("\r\n");
    _exit((int)mcause);
}

static inline void enable_vector_state(void) {
    // mstatus.VS[10:9] must be non-zero to execute vector instructions.
    // Set VS=Dirty (3) so the first RVV instruction won't trap.
    uintptr_t mstatus;
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9);
    __asm__ volatile("csrw mstatus, %0" :: "r"(mstatus));
}

static void crlf(void) { print_uart("\r\n"); }

static void print_title(const char *name) {
    print_uart("\r\n-- ");
    print_uart(name);
    print_uart(" --\r\n");
}

static uint32_t f32_bits(float x) {
    union { float f; uint32_t u; } v = { .f = x };
    return v.u;
}

static void __attribute__((noinline))  print_passfail(const char *what, int ok) {
    print_uart(what);
    print_uart(ok ? ": PASS\r\n" : ": FAIL\r\n");
}

static void __attribute__((noinline)) debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        __asm__ volatile("nop");
    }
}

#ifndef RVV_VSE32_WORKAROUND_OFFSET_FLOATS
#define RVV_VSE32_WORKAROUND_OFFSET_FLOATS 4
#endif

// Work around platforms where vse32 to fully aligned addresses can drop the first beat.
static void __attribute__((noinline)) safe_vse32_f32m1(float *dst, vfloat32m1_t v, size_t vl) {
    static float scratch[RVV_VSE32_WORKAROUND_OFFSET_FLOATS + 64] __attribute__((aligned(64)));
    float *tmp = scratch + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;
    __riscv_vse32_v_f32m1(tmp, v, vl);
    for (size_t i = 0; i < vl; i++) dst[i] = tmp[i];
}

static void __attribute__((noinline)) safe_vse32_i32m1(int32_t *dst, vint32m1_t v, size_t vl) {
    static int32_t scratch[RVV_VSE32_WORKAROUND_OFFSET_FLOATS + 64] __attribute__((aligned(64)));
    int32_t *tmp = scratch + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;
    __riscv_vse32_v_i32m1(tmp, v, vl);
    for (size_t i = 0; i < vl; i++) dst[i] = tmp[i];
}

// RVV: 向量 load -> store -> 从内存读回 -> 打印
// 注意：不要把 RVV intrinsic 直接写在 main() 里，否则 GCC 可能在 main 的序言里
// 先读 vlenb / 做栈调整，导致在 enable_vector_state() 之前就触发 illegal instruction。
static void __attribute__((noinline)) test_vector_load_store_readback_section(void) {
    //print_title("RVV load/store readback print_uart_float");

    
    float temp1[16] __attribute__((aligned(64)));
    for(int i = 0; i < 16; i++) temp1[i] = 17;
    const float src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    size_t vl = __riscv_vsetvl_e32m1(16);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
    safe_vse32_f32m1(temp1, vx, vl);
    //__asm__ volatile("fence rw,rw" ::: "memory");
    debug_delay_cycles(10000);

    //print_uart("vec->mem: ");
    for (int i = 0; i < 16; i++) {
        print_uart_float(temp1[i]);
        write_serial(i == 15 ? '\n' : ' ');
    }

    // int ok = 1;
    // for (int i = 0; i < 16; i++) {
    //     if (f32_bits(temp1[i]) != f32_bits(src[i])) { ok = 0; break; }
    // }
    // print_passfail("readback equals src", ok);
}

// RVV(u64): 向量 load -> store -> 用 print_uart_int_dec 逐个打印
static void __attribute__((noinline)) test_vector_u64_to_print_uart_int_dec_section(void) {
    print_title("RVV u64 -> print_uart_int_dec");

    // e64m1 的 VLMAX 取决于 VLEN；例如 VLEN=512 时 VLMAX=8。
    // 用固定 8 元素，避免 vsetvl 返回 8 而我们却打印/比较 16 导致读到未初始化尾部。
    uint64_t temp[8] __attribute__((aligned(64)));
    const uint64_t src[8] = {
        0ull,
        1ull,
        42ull,
        123456789ull,
        4000000000ull,
        9223372036854775807ull,
        18446744073709551615ull,
        1000000000000ull
    };

    size_t vl = __riscv_vsetvl_e64m1(8);
    vuint64m1_t vx = __riscv_vle64_v_u64m1(src, vl);
    __riscv_vse64_v_u64m1(temp, vx, vl);
    //__asm__ volatile("fence rw,rw" ::: "memory");
    debug_delay_cycles(1000000);

    print_uart("vec->mem(dec): ");
    for (size_t i = 0; i < vl; i++) {
        print_uart_int_dec(temp[i]);
        write_serial(i + 1 == vl ? '\n' : ' ');
    }

    int ok = 1;
    for (size_t i = 0; i < vl; i++) {
        if (temp[i] != src[i]) { ok = 0; break; }
    }
    print_passfail("u64 readback equals src", ok);
}

static void __attribute__((noinline)) test_v1_to_f32_section(void) {
    print_title("v1_to_f32(vfloat32m1_t)");
    float temp0[16] __attribute__((aligned(64)));

    size_t vl = __riscv_vsetvl_e32m1(16);
    vfloat32m1_t v = __riscv_vfmv_v_f_f32m1(1.25f, vl);
    debug_delay_cycles(1000000);
    safe_vse32_f32m1(temp0, v, vl);
    for (int i = 0; i < 16; i++) {
        print_uart_float(temp0[i]);
        write_serial(i == 15 ? '\n' : ' ');
    }

    float out = v1_to_f32(v);
    

    print_uart("v1_to_f32(1.25) => ");
    print_float_fixed3(out);
    crlf();
    print_passfail("bitwise equals 1.25", f32_bits(out) == f32_bits(1.25f));
}

// RVV(f32): 向量 load -> store -> 用 print_float_fixed3 逐个打印
static void __attribute__((noinline)) test_vector_f32_to_print_float_fixed3_section(void) {
    print_title("RVV f32 -> print_float_fixed3");

    float temp2[16] __attribute__((aligned(64)));
    const float src[16] = {
        0.0f, -0.0f, 1.0f, -2.5f,
        3.141592f, 10.0f, 1234.125f, -99999.75f,
        1.0e-3f, -1.0e-3f, 1.0e6f, -1.0e6f,
        0.333333f, -0.666667f, 42.0f, -42.0f
    };

    size_t vl = __riscv_vsetvl_e32m1(16);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
    safe_vse32_f32m1(temp2, vx, vl);
    //__asm__ volatile("fence rw,rw" ::: "memory");
    debug_delay_cycles(1000000);

    for (int i = 0; i < 16; i++) {
        print_uart("temp[");
        print_dec32(i);
        print_uart("]=");
        print_float_fixed3(temp2[i]);
        print_uart("\r\n");
    }

    int ok = 1;
    for (int i = 0; i < 16; i++) {
        if (f32_bits(temp2[i]) != f32_bits(src[i])) { ok = 0; break; }
    }
    print_passfail("f32 readback equals src", ok);
}

// RVV(i32,f32): 索引与数值向量存回内存后，用 print_idx_prefix 打印前缀，再用 print_float_fixed3 打印值
static void __attribute__((noinline)) test_vector_idx_prefix_with_values_section(void) {
    print_title("RVV idx -> print_idx_prefix + value");

    int32_t idx_mem[8] __attribute__((aligned(64)));
    float val_mem[8] __attribute__((aligned(64)));

    const int32_t idx_src[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const float val_src[8] = {1.0f, -1.0f, 2.5f, -2.5f, 3.0f, 10.25f, 123.5f, -999.0f};

    size_t vl = __riscv_vsetvl_e32m1(8);
    vint32m1_t vidx = __riscv_vle32_v_i32m1(idx_src, vl);
    vfloat32m1_t vval = __riscv_vle32_v_f32m1(val_src, vl);
    safe_vse32_i32m1(idx_mem, vidx, vl);
    safe_vse32_f32m1(val_mem, vval, vl);
    //__asm__ volatile("fence rw,rw" ::: "memory");
    debug_delay_cycles(1000000);

    for (int i = 0; i < 8; i++) {
        print_idx_prefix("v", idx_mem[i]);
        print_float_fixed3(val_mem[i]);
        print_uart("\r\n");
    }

    int ok = 1;
    for (int i = 0; i < 8; i++) {
        if (idx_mem[i] != idx_src[i]) { ok = 0; break; }
        if (f32_bits(val_mem[i]) != f32_bits(val_src[i])) { ok = 0; break; }
    }
    print_passfail("idx/value readback equals src", ok);
}

int main(void) {
    // init_uart / write_serial / print_uart
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    print_uart("\r\n==== uart_helper_test ===="
               "\r\n(Each section exercises one function)\r\n");

    // ---------------------------------------------------------------------
    print_title("init_uart / write_serial / print_uart");
    print_uart("print_uart says hello\r\n");
    print_uart("write_serial chars: ");
    write_serial('O');
    write_serial('K');
    crlf();

    // ---------------------------------------------------------------------
    // print_title("print_uart_int_dec(uint64_t)");
    // print_uart("0 => ");
    // print_uart_int_dec(0);
    // crlf();
    // print_uart("42 => ");
    // print_uart_int_dec(42);
    // crlf();
    // print_uart("18446744073709551615 => ");
    // print_uart_int_dec(18446744073709551615ull);
    // crlf();
    // // RVV readback + print (keep RVV out of main prologue; enable VS before calling)
    // enable_vector_state();
    // test_vector_u64_to_print_uart_int_dec_section();

    // ---------------------------------------------------------------------
    // print_title("print_uart_float(float)");
    // print_uart("3.141592 => ");
    // print_uart_float(3.141592f);
    // crlf();
    // print_uart("-2.5 => ");
    // print_uart_float(-2.5f);
    // crlf();
    // // NaN / Inf
    // union { uint32_t u; float f; } q;
    // q.u = 0x7FC00000u;
    // print_uart("nan => ");
    // print_uart_float(q.f);
    // crlf();
    // q.u = 0x7F800000u;
    // print_uart("inf => ");
    // print_uart_float(q.f);
    // crlf();
    // q.u = 0xFF800000u;
    // print_uart("-inf => ");
    // print_uart_float(q.f);
    // crlf();

    // ---------------------------------------------------------------------
    // print_title("my_fabs(float)");
    // float a = -1.25f;
    // float b = my_fabs(a);
    // // Use bit compare to avoid any formatting differences
    // print_passfail("my_fabs(-1.25) == 1.25", f32_bits(b) == f32_bits(1.25f));

    // ---------------------------------------------------------------------
    // print_title("print_uart_hex(unsigned long long)");
    // print_uart("0x0 => ");
    // print_uart_hex(0ull);
    // crlf();
    // print_uart("0x1234 => ");
    // print_uart_hex(0x1234ull);
    // crlf();
    // print_uart("0xFEDCBA9876543210 => ");
    // print_uart_hex(0xFEDCBA9876543210ull);
    // crlf();

    // // ---------------------------------------------------------------------
    // print_title("uart_putc(char)");
    // print_uart("uart_putc: ");
    // uart_putc('X');
    // uart_putc('Y');
    // uart_putc('Z');
    // crlf();

    // ---------------------------------------------------------------------
    // print_title("print_dec32(int32_t)");
    // print_uart("0 => ");
    // print_dec32(0);
    // crlf();
    // print_uart("123456 => ");
    // print_dec32(123456);
    // crlf();
    // print_uart("-123456 => ");
    // print_dec32(-123456);
    // crlf();
    // Note: avoid INT32_MIN because this helper does v=-v (would overflow)

    // ---------------------------------------------------------------------
    // print_title("print_float_fixed3(float)");
    // print_uart("0 => ");
    // print_float_fixed3(0.0f);
    // crlf();
    // print_uart("-0 => ");
    // union { uint32_t u; float f; } nz = { .u = 0x80000000u };
    // print_float_fixed3(nz.f);
    // crlf();
    // print_uart("1.234567 => ");
    // print_float_fixed3(1.234567f);
    // crlf();
    // print_uart("-2.5 => ");
    // print_float_fixed3(-2.5f);
    // crlf();
    // print_uart("1e10 (sci) => ");
    // print_float_fixed3(1.0e10f);
    // crlf();
    // // NaN/Inf
    // q.u = 0x7FC00000u;
    // print_uart("nan => ");
    // print_float_fixed3(q.f);
    // crlf();
    // q.u = 0x7F800000u;
    // print_uart("inf => ");
    // print_float_fixed3(q.f);
    // crlf();
    // q.u = 0xFF800000u;
    // print_uart("-inf => ");
    // print_float_fixed3(q.f);
    // crlf();

    // ---------------------------------------------------------------------
    // print_title("print_f32_scalar(const char*, float)");
    // print_f32_scalar("val", 3.5f);

    // ---------------------------------------------------------------------
    enable_vector_state();
    //test_vector_load_store_readback_section();
    test_vector_f32_to_print_float_fixed3_section();
    test_vector_idx_prefix_with_values_section();
    //test_v1_to_f32_section();

    // ---------------------------------------------------------------------
    // print_title("print_idx_prefix(const char*, int)");
    // print_idx_prefix("a", 7);
    // print_uart("<value would follow>\r\n");

    // ---------------------------------------------------------------------
    print_title("print_f32_array(const char*, const float*, int)");
    {
        static const float arr[4] = { 0.0f, 1.0f, -2.5f, 1234.125f };
        print_f32_array("arr", arr, 4);
    }

    print_uart("\r\nDONE\r\n");
    return 0;
}
