#include "matrix_kernel_1230.h"
#define MK_MMAX 64
float matrix_kernel_debug[MK_MMAX] __attribute__((aligned(64), section(".uncached_buffer")));
// ---------------------------------------------------------------------------
// Strided variant: allows A/B/C row stride to be specified explicitly (bytes).
// This is useful to avoid scalar "packing" loops when the source tensors are
// already laid out row-major but with a row stride != k.
//
// A: base pointer to row 0, row stride = lda_bytes
// B: base pointer to row 0, row stride = ldb_bytes
// C: base pointer to row 0, row stride = ldc_bytes
int matrix_kernel_matmul_i8_i32_abt_strided(const int8_t *A, int lda_bytes,
                                                          const int8_t *B, int ldb_bytes,
                                                          int32_t *C, int ldc_bytes,
                                                          int m, int n, int k) {
    // if (1) {
    //     print_uart("\r\n[debug_trace] Enter abt_strided\r\n");
    //     print_uart("  A="); print_uart_hex((uint64_t)(uintptr_t)A);
    //     print_uart(" lda="); print_uart_int_dec((uint64_t)lda_bytes);
    //     print_uart("\r\n");
    //     print_uart("  B="); print_uart_hex((uint64_t)(uintptr_t)B);
    //     print_uart(" ldb="); print_uart_int_dec((uint64_t)ldb_bytes);
    //     print_uart("\r\n");
    //     print_uart("  m="); print_uart_int_dec((uint64_t)m);
    //     print_uart(" n="); print_uart_int_dec((uint64_t)n);
    //     print_uart(" k="); print_uart_int_dec((uint64_t)k);
    //     print_uart("\r\n");
    // }

    int tile_m = 0, tile_n = 0, tile_k = 0;
    //int tile_m_pad = 0, tile_n_pad = 0, tile_k_pad = 0;

    for (int i = 0; i < m; i += tile_m) {
        tile_m = matrix_kernel_msettilem(m - i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = matrix_kernel_msettilen(n - j);
            if (tile_n <= 0) {
                 print_uart("Panic: tile_n=0\r\n");
                 break;
            }

            // if (1) { // Force debug print for diagnosis
            //     print_uart("[matmul_debug] i="); print_uart_int_dec((uint64_t)i);
            //     print_uart(" tile_m="); print_uart_int_dec((uint64_t)tile_m);
            //     print_uart(" j="); print_uart_int_dec((uint64_t)j);
            //     print_uart(" tile_n="); print_uart_int_dec((uint64_t)tile_n);
            //     print_uart(" n="); print_uart_int_dec((uint64_t)n);
            //     print_uart("\r\n");
            // }
            int32_t *cptr = (int32_t *)((uint8_t *)C + (size_t)i * (size_t)ldc_bytes) + j;

            // if (1) {
            //     print_uart("  [trace] i="); print_uart_int_dec((uint64_t)i);
            //     print_uart(" j="); print_uart_int_dec((uint64_t)j);
            //     print_uart(" tile_n="); print_uart_int_dec((uint64_t)tile_n);
            //     print_uart("\r\n");
            // }

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            static uint64_t s_strided_calls = 0;
            const uint64_t dbg_id = ++s_strided_calls;
            if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
                matrix_kernel_debug_print_prefix();
                print_uart("abt_strided id=");
                matrix_kernel_debug_print_u64(dbg_id);
                print_uart(" m=");
                matrix_kernel_debug_print_u64((uint64_t)m);
                print_uart(" n=");
                matrix_kernel_debug_print_u64((uint64_t)n);
                print_uart(" k=");
                matrix_kernel_debug_print_u64((uint64_t)k);
                print_uart(" lda=");
                matrix_kernel_debug_print_u64((uint64_t)lda_bytes);
                print_uart(" ldb=");
                matrix_kernel_debug_print_u64((uint64_t)ldb_bytes);
                print_uart(" ldc=");
                matrix_kernel_debug_print_u64((uint64_t)ldc_bytes);
                print_uart("\r\n");

                matrix_kernel_debug_print_prefix();
                print_uart(" i=");
                matrix_kernel_debug_print_u64((uint64_t)i);
                print_uart(" tile_m=");
                matrix_kernel_debug_print_u64((uint64_t)tile_m);
                print_uart(" j=");
                matrix_kernel_debug_print_u64((uint64_t)j);
                print_uart(" tile_n=");
                matrix_kernel_debug_print_u64((uint64_t)tile_n);
                print_uart(" cptr=");
                matrix_kernel_debug_print_ptr(cptr);
                print_uart("\r\n");
            }
#endif
            matrix_kernel_mlce32_acc0((const int32_t *)cptr, ldc_bytes);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = matrix_kernel_msettilek(k - kk);
                //tile_k_pad = matrix_kernel_msettilek(8);
                const int8_t *aptr = (const int8_t *)((const uint8_t *)A + (size_t)i * (size_t)lda_bytes) + kk;
                // B is provided as (n x k) row-major, with explicit row stride.
                const int8_t *bptr = (const int8_t *)((const uint8_t *)B + (size_t)j * (size_t)ldb_bytes) + kk;

                // if (1) {
                //     print_uart("    [trace] kk="); print_uart_int_dec((uint64_t)kk);
                //     print_uart(" aptr="); print_uart_hex((uint64_t)(uintptr_t)aptr);
                //     print_uart(" bptr="); print_uart_hex((uint64_t)(uintptr_t)bptr);
                //     print_uart("\r\n");
                // }

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
                if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
                    matrix_kernel_debug_print_prefix();
                    print_uart("  kk=");
                    matrix_kernel_debug_print_u64((uint64_t)kk);
                    print_uart(" tile_k=");
                    matrix_kernel_debug_print_u64((uint64_t)tile_k);
                    print_uart(" aptr=");
                    matrix_kernel_debug_print_ptr(aptr);
                    print_uart(" bptr=");
                    matrix_kernel_debug_print_ptr(bptr);
                    print_uart("\r\n");

                    matrix_kernel_debug_print_prefix();
                    matrix_kernel_debug_print_arr_i8("  Arow", aptr, (tile_k > 0 ? tile_k : 0));
                    matrix_kernel_debug_print_prefix();
                    matrix_kernel_debug_print_arr_i8("  Brow", bptr, (tile_k > 0 ? tile_k : 0));
                }
#endif
                matrix_kernel_mlae8_tr0(aptr, lda_bytes);
                matrix_kernel_mlbe8_tr1(bptr, ldb_bytes);
                matrix_kernel_mqma_b_acc0_tr0_tr1();
            }

            matrix_kernel_msce32_acc0(cptr, ldc_bytes);
            debug_delay_cycles(200);
        }
    }
    debug_delay_cycles(100);

    return 0;
}
