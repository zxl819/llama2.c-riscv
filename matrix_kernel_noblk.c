#include "matrix_kernel_noblk_1231.h"

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

        // Pack A (W rows) into contiguous [d x count]
        // Use a VLA - keep it reasonably small in test scenarios.
        __attribute__((aligned(64))) int8_t a_pack[d * count];
        for (int r = 0; r < d; r++) {
            const int8_t *src = wq + (ptrdiff_t)r * (ptrdiff_t)w_row_stride + (ptrdiff_t)base;
            int8_t *dst = a_pack + (size_t)r * (size_t)count;
            for (int c = 0; c < count; c++) dst[c] = src[c];
        }

        // B segment (single row) for n=1 output
        const int8_t *Bseg = xq + base;

        // Allocate contiguous C buffer [d] and clear it.
        __attribute__((aligned(64))) int32_t c_pack[d];
        matrix_kernel_rvv_zero_i32(c_pack, d);

        // Call the non-strided abt path with n=1; m tiling is internal.
        (void)matrix_kernel_matmul_i8_i32_abt(a_pack, Bseg, c_pack, d, 1, count);
        debug_delay_cycles(2000);
        /* Ensure matrix unit writes are visible to subsequent CPU loads. */
        //__asm__ volatile("fence rw, rw" ::: "memory");

        // Materialize ws for these rows (contiguous)
        // Use a VLA float buffer for ws per-row
        __attribute__((aligned(64))) float ws_local[d];
        for (int r = 0; r < d; r++) {
            ws_local[r] = ws[(ptrdiff_t)r * (ptrdiff_t)groups_per_row + (ptrdiff_t)g];
        }

        // RVV-accelerated accumulation: xout[0:d] += float(c_pack[0:d]) * ws_local[0:d] * xs[g]
        matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
            xout,
            c_pack,
            ws_local,
            xs[g],
            d
        );
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

        // Allocate C buffer [batch * d]
        __attribute__((aligned(64))) int32_t c_pack[d * batch];
        matrix_kernel_rvv_zero_i32(c_pack, d * batch);

        // Call ABT: A=w_pack (n=d, k=count),B=x_pack (m=batch, k=count).
        // Result: d x batch .
        (void)matrix_kernel_matmul_i8_i32_abt(x_pack, w_pack, c_pack, batch, d, count);
        debug_delay_cycles(2000);
        //__asm__ volatile("fence rw, rw" ::: "memory");

        // Prepare ws_local for this group (shared across batches)
        __attribute__((aligned(64))) float ws_local[d];
        for (int r = 0; r < d; r++) {
            ws_local[r] = ws[(ptrdiff_t)r * (ptrdiff_t)groups_per_row + (ptrdiff_t)g];
        }

        // Loop over batch rows to accumulate
        for (int b = 0; b < batch; b++) {
            // c_pack is (d, batch). Column b is at c_pack[0*batch + b], c_pack[1*batch + b], ...
            const int32_t *c_src = c_pack + b * d;
            //ptrdiff_t c_stride = (ptrdiff_t)batch * sizeof(int32_t);

            float x_scale = xs[b * groups_per_row + g];
            float *xout_ptr = xout + b * d; // xout is (batch, d) row-major
            
            matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
                xout_ptr,
                c_src,
                ws_local,
                x_scale,
                d
            );
        }
    }
}
