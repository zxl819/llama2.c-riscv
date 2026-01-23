#include "uart_helper.c"
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
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

static void __attribute__((noinline)) debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        __asm__ volatile("nop");
    }
}
static void __attribute__((noinline)) test_vector_load_store(void) {
    print_title("RVV load/store varying vl");

    
    float temp1[2048] __attribute__((aligned(64)));
    int step = 0;
    for(int i = 0; i < 16; i++) temp1[i] = 17;
    const float src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    for(int ii = 0; ii < 16; ii++) {
    size_t vl = __riscv_vsetvl_e32m1(ii);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
    __riscv_vse32_v_f32m1(&temp1[step], vx, vl);
    debug_delay_cycles(1000);
    

    //print_uart("vec->mem: ");
    for (int i = 0; i < vl; i++) {
        print_float_fixed3(temp1[i]);
        write_serial(i == vl ? '\n' : ' ');
    }

    int ok = 1;
    for (int i = 0; i < vl; i++) {
        if (f32_bits(temp1[i]) != f32_bits(src[i])) { ok = 0; break; }
    }
    print_passfail("readback equals src", ok);
    step += vl;
}
}

int main(void) {
    // init_uart / write_serial / print_uart
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    print_uart("\r\n==== uart_helper_test ===="
               "\r\n(Each section exercises one function)\r\n");
    enable_vector_state();
    test_vector_load_store();
    print_uart("\r\nDONE\r\n");

}