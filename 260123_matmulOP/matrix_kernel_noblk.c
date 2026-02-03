// ---
// author: zhaoxinlei
// version: 1.0
// ---
// [代码说明]
// 本文件实现了基于矩阵扩展的优化量化矩阵乘法 (qmatmul) 内核，避免了块阻塞 (noblk variant)。
// 关键优化点：
// 1. 使用静态分配于 .uncached_buffer 段的中间缓冲区 c_pack_static，解决矩阵单元与 CPU 之间的缓存一致性问题。
// 2. 支持块缩放 (Block-scaling) 逻辑：输入 x 和权重 w 每 32 个元素 (GS=32) 拥有独立的缩放因子。
// 3. 核心计算调用 matrix_kernel_matmul_i8_i32_abt_strided 充分利用硬件矩阵指令。
// 4. 使用 RVV (Vector) 指令进行后处理，将 int32 的累加结果乘以两个缩放因子并累加回 float 输出数组。

#include "matrix_kernel_noblk_1231.h"

#define MAX_BATCH_C_PACK_SIZE (2048 * 16) 
__attribute__((aligned(64))) static int32_t c_pack_batch_buf[MAX_BATCH_C_PACK_SIZE];

void matrix_kernel_qmatmul_f32_noblk(float *xout,
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

        // Allocate contiguous C buffer [d] and clear it.
        __attribute__((aligned(64))) int32_t c_pack[d];
        matrix_kernel_rvv_zero_i32(c_pack, d);

        // Call the strided abt path with n=1; m tiling is internal.
        // A (W) is d x count. Stride = w_row_stride.
        // B (X) is 1 x count. Stride = count (logical).
        // C is d x 1. Stride = sizeof(int32_t) = 4.
        (void)matrix_kernel_matmul_i8_i32_abt_strided(  A_ptr, w_row_stride, 
                                                        B_ptr, count, 
                                                        c_pack, sizeof(int32_t), 
                                                        d, 1, count);
        
        debug_delay_cycles(800);
       

        // Calc pointer to ws[0, g] - Start address = ws + 0*row_stride + g
        const float *ws_base_ptr = ws + g;
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

void matrix_kernel_qmatmul_f32_noblk_batch(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride,
                                      int batch) {
    if (n == 0 || d == 0 || batch == 0) return;

    const int num_groups = (n + gs - 1) / gs;
    const int groups_per_row = n / gs;

    // Zero entire output: xout is size (d * batch)
    matrix_kernel_rvv_zero_f32(xout, d * batch);

    for (int g = 0; g < num_groups; g++) {
        const int base = g * gs;
        const int count = (base + gs <= n) ? gs : (n - base);

        // Pack B (W rows) into contiguous [d x count]
        /*
        __attribute__((aligned(64))) int8_t w_pack[d * count];
        for (int r = 0; r < d; r++) {
            const int8_t *src = wq + (ptrdiff_t)r * (ptrdiff_t)w_row_stride + (ptrdiff_t)base;
            int8_t *dst = w_pack + (size_t)r * (size_t)count;
            for (int c = 0; c < count; c++) dst[c] = src[c];
        }

        // Pack A (X rows) into contiguous [batch x count]
        __attribute__((aligned(64))) int8_t x_pack[batch * count];
        for(int b=0; b<batch; ++b) {
             const int8_t *src = xq + b * n + base;
             int8_t *dst = x_pack + b * count;
             for(int c=0; c<count; ++c) dst[c] = src[c];
        }
        */

        const int8_t *A_addr = xq + base;
        const int8_t *B_addr = wq + base;

        // Use global uncached buffer instead of stack VLA
        if ((d * batch) > MAX_BATCH_C_PACK_SIZE) {
            print_uart("Error: c_pack_batch_buf overflow\r\n");
            return;
        }
        int32_t *c_pack = c_pack_batch_buf;
        
        // debug_delay_cycles(5000);
        matrix_kernel_rvv_zero_i32(c_pack, d * batch);

        // Call ABT: A=x (m=batch, k=count), B=w (n=d, k=count).
        // Result: d x batch .
        
        // A (X) stride = n (input row width)
        // B (W) stride = w_row_stride
        // C stride = d * 4 (row width of result C is d elements)
        
        // Safety check for stack overflow
        // if (1) {
        //     print_uart("[noblk_batch] g="); print_uart_int_dec((uint64_t)g);
        //     print_uart(" d="); print_uart_int_dec((uint64_t)d);
        //     print_uart(" batch="); print_uart_int_dec((uint64_t)batch);
        //     print_uart(" A_addr="); print_uart_hex((uint64_t)(uintptr_t)A_addr);
        //     print_uart(" B_addr="); print_uart_hex((uint64_t)(uintptr_t)B_addr);
        //     print_uart(" C_addr="); print_uart_hex((uint64_t)(uintptr_t)c_pack);
        //     print_uart("\r\n");
        // }

        (void)matrix_kernel_matmul_i8_i32_abt_strided(  A_addr, n, 
                                                        B_addr, w_row_stride, 
                                                        c_pack, d * sizeof(int32_t),
                                                        batch, d, count);
            // print_uart("[noblk_batch] g="); print_uart_int_dec((uint64_t)g);
            // print_uart(" d="); print_uart_int_dec((uint64_t)d);
            // print_uart(" batch="); print_uart_int_dec((uint64_t)batch);
            // print_uart("\r\n");
        debug_delay_cycles(800);
        //print_uart("matmul done\r\n"); 
        //__asm__ volatile("fence rw, rw" ::: "memory");

        // Prepare ws_local for this group (shared across batches)
        const float *ws_base_ptr = ws + g;

        // Loop over batch rows to accumulate
        for (int b = 0; b < batch; b++) {
            // c_pack is (d, batch). Column b is at c_pack[0*batch + b], c_pack[1*batch + b], ...
            const int32_t *c_src = c_pack + b * d;
            //ptrdiff_t c_stride = (ptrdiff_t)batch * sizeof(int32_t);

            float x_scale = xs[b * groups_per_row + g];
            float *xout_ptr = xout + b * d; // xout is (batch, d) row-major
            debug_delay_cycles(100);
            // print_uart("pointer done\r\n"); 
            matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
            xout_ptr,
            c_src,
            ws_base_ptr,
            groups_per_row, // stride in floats
            x_scale,
            d
        );
        debug_delay_cycles(100);
            // matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
            //     xout_ptr,
            //     c_src,
            //     ws_base_ptr,
            //     groups_per_row,
            //     x_scale,
            //     d
            // );
            // print_uart("rvv accum done\r\n"); 
        }
    }
}
