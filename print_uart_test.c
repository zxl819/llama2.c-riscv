// Standalone bare-metal smoke test for print_uart.
//
// Build & run on Spike (tohost printf backend):
//   make rvrunbare RV_BARE_APP=print_uart_test.c BARE_EMBED_BLOBS=0 PRINT_WAY=printf
//
// Build for hardware UART backend:
//   make rvbare RV_BARE_APP=print_uart_test.c BARE_EMBED_BLOBS=0 PRINT_WAY=uart

#include <stdint.h>
#include <riscv_vector.h>

// Intentionally include as a .c to match existing test style.
#include "uart_helper.c"

#ifndef CLOCK_FREQUENCY
#define CLOCK_FREQUENCY 25000000u
#endif

#ifndef UART_BITRATE
#define UART_BITRATE 115200u
#endif

static void print_banner(void) {
    print_uart("\r\n==== print_uart_test ===="
               "\r\n");
}

int main(void) {
    // In PRINT_WAY=printf mode, init_uart is a no-op shim.
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    print_banner();

    print_uart("[1] basic string: Hello UART\r\n");
    print_uart("[2] empty string: ");
    print_uart("");
    print_uart("<end>\r\n");

    print_uart("[3] special chars: ");
    write_serial('A');
    write_serial(' ');
    write_serial('0');
    write_serial(' ');
    write_serial('#');
    print_uart("\r\n");

    print_uart("[4] long line: ");
    print_uart("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\r\n");

    print_uart("[5] hex helper (should be 0000000000001234): ");
    print_uart_hex(0x1234ull);
    print_uart("\r\n");

    print_uart("DONE\r\n");
    return 0;
}
