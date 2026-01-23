#include <riscv_vector.h>
#include "uart_helper.c"
#include <string.h>
#include <stdint.h>

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200
#define RVV_PADDING_STATIC 1
#define RVV_PADDING_NO_ALLOCATOR 1
#include "RVV_padding.c"
//1843200 
static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}

static inline void fill_float(float *buf, size_t count, float value) {
    for (size_t i = 0; i < count; i++) buf[i] = value;
}

static void  __attribute__((noinline)) print_floats(const char *tag, const float *buf, size_t count) {
    print_uart(tag);
    for (size_t i = 0; i < count; i++) {
        print_uart_float(buf[i]);
        print_uart(" ");
        if (((i + 1) % 8) == 0) print_uart("\r\n");
    }
    if ((count % 8) != 0) print_uart("\r\n");
}

typedef enum {
    STORE_VSE32 = 0,
} store_mode_t;

// Some platforms exhibit a hardware issue where vse32 to certain destination alignments
// drops the first beat(s). If you cannot access vstart CSR on your hardware, this test can
// still diagnose/store-workaround by routing the vector store to a known-good scratch address
// and scalar-copying into the real destination.
#ifndef VSE32_SAFE_STORE
#define VSE32_SAFE_STORE 0
#endif

// Choose a scratch offset that avoids the failing store path (in floats).
// Based on common observations, +3 or +4 are good candidates.
#ifndef VSE32_SAFE_SCRATCH_OFFSET_FLOATS
#define VSE32_SAFE_SCRATCH_OFFSET_FLOATS 0
#endif

// Optional experiment: split a single vse32(vl) into multiple vse32 stores.
// This helps distinguish "first beat of a vector store is dropped" from other issues.
#ifndef VSE32_SPLIT_STORE
#define VSE32_SPLIT_STORE 0
#endif

// Optional sweep to map which destination offsets trigger the issue.
// When enabled, runs offsets starting at 0.
#ifndef VSE32_ENABLE_SWEEP
#define VSE32_ENABLE_SWEEP 0
#endif

// Maximum offset (in floats) to try during sweep.
#ifndef VSE32_SWEEP_MAX_OFFSET_FLOATS
#define VSE32_SWEEP_MAX_OFFSET_FLOATS 16
#endif

// Store chunk size (in elements) when VSE32_SPLIT_STORE=1.
// For vl=16, the default results in 8 + 8.
#ifndef VSE32_SPLIT_CHUNK
#define VSE32_SPLIT_CHUNK 16
#endif

static __attribute__((aligned(64))) float g_vse32_scratch[VSE32_SAFE_SCRATCH_OFFSET_FLOATS + 64];

static inline void vse32_store_maybe_safe(float *dst_ptr, vfloat32m1_t v, size_t vl) {
#if VSE32_SAFE_STORE
    float *tmp = g_vse32_scratch;
    __riscv_vse32_v_f32m1(tmp + VSE32_SAFE_SCRATCH_OFFSET_FLOATS, v, vl);
    //asm volatile("fence rw, rw" ::: "memory");
    for (size_t i = 0; i < vl; i++) dst_ptr[i] = tmp[VSE32_SAFE_SCRATCH_OFFSET_FLOATS + i];
#else
    __riscv_vse32_v_f32m1(dst_ptr, v, vl);
    debug_delay_cycles(100);
#endif
}

static inline void vse32_store_from_src(float *dst_ptr, const float *src_ptr, size_t vl) {
#if VSE32_SPLIT_STORE
    size_t offset = 0;
    while (offset < vl) {
        size_t chunk = (vl - offset) < (size_t)VSE32_SPLIT_CHUNK ? (vl - offset) : (size_t)VSE32_SPLIT_CHUNK;
        vfloat32m1_t v = __riscv_vle32_v_f32m1(src_ptr + offset, chunk);
        vse32_store_maybe_safe(dst_ptr + offset, v, chunk);
        offset += chunk;
    }
#else
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src_ptr, vl);
    vse32_store_maybe_safe(dst_ptr, v, vl);
#endif
}

static void run_case(const char *case_name, store_mode_t mode, float *dst_base, size_t dst_count, float *dst_ptr, const float *src, size_t vl) {
    print_uart("\r\n== ");
    print_uart(case_name);
    print_uart(" (mode=");
    (void)mode;
    print_uart("vse32");
    print_uart(")");
    print_uart(" ==\r\n");

    fill_float(dst_base, dst_count, 1111.0f);

    print_uart("dst_base=0x");
    print_uart_hex((unsigned long long)(uintptr_t)dst_base);
    print_uart(" dst_ptr=0x");
    print_uart_hex((unsigned long long)(uintptr_t)dst_ptr);
    print_uart(" (offset_floats=");
    print_uart_int_dec((int)(dst_ptr - dst_base));
    print_uart(") addr_mod64=");
    print_uart_int_dec((int)((uintptr_t)dst_ptr & 63u));
    print_uart(" addr_mod16=");
    print_uart_int_dec((int)((uintptr_t)dst_ptr & 15u));
    print_uart("\r\n");

    print_uart("cfg: SAFE_STORE=");
    print_uart_int_dec((int)VSE32_SAFE_STORE);
    print_uart(" SAFE_SCR_OFF=");
    print_uart_int_dec((int)VSE32_SAFE_SCRATCH_OFFSET_FLOATS);
    print_uart(" SPLIT_STORE=");
    print_uart_int_dec((int)VSE32_SPLIT_STORE);
    print_uart(" SPLIT_CHUNK=");
    print_uart_int_dec((int)VSE32_SPLIT_CHUNK);
    print_uart("\r\n");

    // Perform the store under test.
    vse32_store_from_src(dst_ptr, src, vl);

    //debug_delay_cycles(50000);

    // Print a window around the beginning of dst_base so we can see if the missing beats move.
    print_floats("dst[0..19]:\r\n", dst_base, 32);

    // Check expected mapping.
    int mismatches = 0;
    for (size_t i = 0; i < vl; i++) {
        float expected = (float)(i + 1);
        float got = dst_ptr[i];
        if (got != expected) {
            mismatches++;
            if (mismatches <= 8) {
                print_uart("Mismatch at region idx ");
                print_uart_int_dec((int)i);
                print_uart(": got=");
                print_uart_float(got);
                print_uart(" expected=");
                print_uart_float(expected);
                print_uart("\r\n");
            }
        }
    }
    if (mismatches == 0) {
        print_uart("Case result: ALL OK\r\n");
    } else {
        print_uart("Case result: mismatches=");
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
    print_uart("\r\n[vse_offset_test] start\r\n");

    size_t vlmax = __riscv_vsetvlmax_e32m1();
    print_uart("vlmax=");
    print_uart_int_dec((int)vlmax);
    print_uart("\r\n");

    // Keep the test simple: assume vlmax is 16 for 512-bit VLEN and e32m1.
    // Still run with whatever vlmax reports.
    __attribute__((aligned(64))) float src[16];
    __attribute__((aligned(64))) float src2[16];
    __attribute__((aligned(64))) float dst1[32];
    __attribute__((aligned(64))) float dst2[32];
    __attribute__((aligned(64))) float dst3[32];
    __attribute__((aligned(64))) float dst4[32];
    __attribute__((aligned(64))) float dst5[32];
    __attribute__((aligned(64))) float dst6[32];
    __attribute__((aligned(64))) float dst7[32];
    __attribute__((aligned(64))) float dst8[32];
    __attribute__((aligned(64))) float dst9[32];
    __attribute__((aligned(64))) float dst10[32];
    __attribute__((aligned(64))) float dst11[32];
    __attribute__((aligned(64))) float dst12[32];

    for (int i = 0; i < 16; i++) src[i] = (float)(i + 1);
    for (int i = 0; i < 16; i++) src2[i] = (float)(i + 2);

    print_floats("src[0..15]:\r\n", src, 16);

    // Sweep offsets to map which dst alignments trigger the issue.
    // Keep within dst[0..31] bounds: require offset + vlmax <= 32.
#if VSE32_ENABLE_SWEEP
    for (int off = 0; off <= (int)VSE32_SWEEP_MAX_OFFSET_FLOATS; off++) {
        if ((size_t)off + vlmax > 32u) break;
        print_uart("\r\n[SWEEP] offset_floats=");
        print_uart_int_dec(off);
        print_uart("\r\n");
        run_case("SWEEP dst1", STORE_VSE32, dst1, 32, &dst1[off], src, vlmax);
    }
#endif

    // Try a few hand-picked offsets (floats) to characterize alignment sensitivity.
    run_case("CASE A: store src -> dst+0", STORE_VSE32, dst1, 32, &dst1[0], src, vlmax);
    run_case("CASE A2: store src2 -> dst+0", STORE_VSE32, dst1, 32, &dst1[0], src2, vlmax);
    run_case("CASE A3: store -> dst+2", STORE_VSE32, dst1, 32, &dst1[2], src, vlmax);
    run_case("CASE B: store -> dst+3", STORE_VSE32, dst1, 32, &dst1[3], src, vlmax);
    run_case("CASE C: store -> dst+4", STORE_VSE32, dst1, 32, &dst1[4], src, vlmax);
    run_case("CASE D: store -> dst+6", STORE_VSE32, dst1, 32, &dst1[6], src, vlmax);
    run_case("CASE E: store -> dst+8", STORE_VSE32, dst1, 32, &dst1[8], src, vlmax);
    run_case("CASE F: store -> dst+10", STORE_VSE32, dst1, 32, &dst1[10], src, vlmax);
    run_case("CASE G: store -> dst+12", STORE_VSE32, dst1, 32, &dst1[12], src, vlmax);
    run_case("CASE H: store -> dst+14", STORE_VSE32, dst1, 32, &dst1[14], src, vlmax);
    run_case("CASE I: store -> dst+16", STORE_VSE32, dst1, 32, &dst1[16], src, vlmax);

    print_uart("\r\n[vse_offset_test] done\r\n");
    while (1) {
        asm volatile("wfi");
    }

    return 0;
}
