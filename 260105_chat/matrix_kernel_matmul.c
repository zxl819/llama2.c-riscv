#include "matrix_kernel_1230.h"

int matrix_kernel_matmul_i8_i32_abt(const int8_t *A, const int8_t *B, int32_t *C,
                                                  int m, int n, int k) {
#if MATRIX_KERNEL_DEBUG_PRINT
//print_uart("=== matrix_kernel_matmul_i8_i32_abt entered ===\r\n");
// debug_delay_cycles(20);
#endif
    const int lda_bytes = k * (int)sizeof(int8_t);
    // B is (n x k) row-major: each row has k elements -> row stride = k bytes
    const int ldb_bytes = k * (int)sizeof(int8_t);
    const int ldc_bytes = n * (int)sizeof(int32_t);

    

    int tile_m = 0, tile_n = 0, tile_k = 0;

    for (int i = 0; i < m; i += tile_m) {
        tile_m = matrix_kernel_msettilem(m - i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = matrix_kernel_msettilen(n - j);
            int32_t *cptr = C + i * n + j;

            #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            // static uint64_t s_strided_calls = 0;
            // const uint64_t dbg_id = ++s_strided_calls;
            // if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
            //     matrix_kernel_debug_print_prefix();
            //     print_uart("abt_strided id=");
            //     matrix_kernel_debug_print_u64(dbg_id);
                //  print_uart("\r\r");
            //    matrix_kernel_debug_print_u64((uint64_t)m);
            //     print_uart(" n=");
            //     matrix_kernel_debug_print_u64((uint64_t)n);
            // //     print_uart(" k=");
            //     matrix_kernel_debug_print_u64((uint64_t)k);
            //     print_uart(" lda=");
            //     matrix_kernel_debug_print_u64((uint64_t)lda_bytes);
            //     print_uart(" ldb=");
            //     matrix_kernel_debug_print_u64((uint64_t)ldb_bytes);
            //     print_uart(" ldc=");
            //     matrix_kernel_debug_print_u64((uint64_t)ldc_bytes);
            //     print_uart("\r\n");

            //     matrix_kernel_debug_print_prefix();
            //     print_uart(" i=");
            //     matrix_kernel_debug_print_u64((uint64_t)i);
            //     print_uart(" tile_m=");
            //     matrix_kernel_debug_print_u64((uint64_t)tile_m);
            //     print_uart(" j=");
            //     matrix_kernel_debug_print_u64((uint64_t)j);
            //     print_uart(" tile_n=");
            //     matrix_kernel_debug_print_u64((uint64_t)tile_n);
                // print_uart(" cptr=");
                // matrix_kernel_debug_print_ptr(cptr);
            //     print_uart("\r\n");

            // }
#endif
            matrix_kernel_mlce32_acc0((const int32_t *)cptr, ldc_bytes);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = matrix_kernel_msettilek(k - kk);
                const int8_t *aptr = A + i * k + kk;
                // B is provided as (n x k) row-major; bptr should point to row j, column kk
                const int8_t *bptr = B + j * k + kk;
// #if MATRIX_KERNEL_DEBUG_PRINT         
//                 print_uart(" bptr=");
//                 matrix_kernel_debug_print_ptr(bptr);
//                 print_uart("\r\n");
// #endif
                matrix_kernel_mlae8_tr0(aptr, lda_bytes);
                matrix_kernel_mlbe8_tr1(bptr, ldb_bytes);
                matrix_kernel_mqma_b_acc0_tr0_tr1();
            }

            matrix_kernel_msce32_acc0(cptr, ldc_bytes);
            debug_delay_cycles(200);
        }
    }
    debug_delay_cycles(50);
    //return 0;
}
