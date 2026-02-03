#include "matrix_kernel_noblk_1231.h"

#define MAX_BATCH_C_PACK_SIZE (2048 * 16) 
__attribute__((aligned(64))) int32_t c_pack_batch_buf[MAX_BATCH_C_PACK_SIZE];
void matrix_kernel_qmatmul_f32_noblk_CT(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride) {
    if (n == 0 || d == 0) return;

    const int num_groups = (n + gs - 1) / gs;
    const int groups_per_row = n / gs;

    // Zero entire output once
    matrix_kernel_rvv_zero_f32(xout, d);

    for (int g = 0; g < num_groups; g++) {
        const int base = g * gs;
        const int count = (base + gs <= n) ? gs : (n - base);

        const int8_t *A_ptr = wq + base;
        const int8_t *B_ptr = xq + base;

        // Use a static buffer in .uncached_buffer to ensure matrix HW writes are visible.
        //__attribute__((aligned(64))) int32_t c_pack[d];
        int32_t *c_pack = c_pack_batch_buf;

        matrix_kernel_rvv_zero_i32(c_pack, d);

        // Call the strided abt path using logic: C^T = B * A^T
        // B (xq) is 1 x count, stride = count
        // A (wq) is d x count, stride = w_row_stride
        // Result C is 1 x d, stride = 4
        (void)matrix_kernel_matmul_i8_i32_abt_strided(  B_ptr, count, 
                                                        A_ptr, w_row_stride, 
                                                        c_pack, sizeof(int32_t), 
                                                        1, d, count);
        
        debug_delay_cycles(800);
        
        // if (1) {
        //      print_uart("[noblk_CT] ws_post="); print_uart_int_dec((uint64_t)(uintptr_t)ws);
        //      print_uart("\r\n");
        // }

        // Calc pointer to ws[0, g] - Start address = ws + 0*row_stride + g
        const float *ws_base_ptr = ws + g;
        //print_uart(" ws_base_ptr="); print_uart_int_dec((uint64_t)(uintptr_t)ws_base_ptr);
        debug_delay_cycles(800);
        // RVV-accelerated accumulation: xout[0:d] += float(c_pack[0:d]) * ws_strided[0:d] * xs[g]
        matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
            xout,
            c_pack,
            ws_base_ptr,
            groups_per_row, // stride in floats
            xs[g],
            d
        );
        debug_delay_cycles(100);
    }
}
 //(ptrdiff_t)sizeof(float), /* ws stride bytes (contiguous) */