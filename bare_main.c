// Forward declare printf provided by bare_syscalls.c
int printf(const char*, ...);
// stdint not strictly needed here; remove to avoid depending on libc headers

// Minimal bare-metal demo: uses riscv-dnn syscalls.c (tohost) printf
// Link with riscv-dnn/include/common/crt.S and include/link.ld via Makefile rvbare

int main(void) {
    printf("Hello, RISC-V bare-metal (Spike tohost)!\n");
    printf("This is a smoke test that your toolchain and linker script work.\n");
    // Return 0 -> syscalls.c will signal tohost exit
    return 0;
}
