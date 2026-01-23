#include "matrix_kernel_noblk_1231.h"

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

        // Call the non-strided abt path with m=1, n=d to produce 1xd continuous output
        // Mathematically: C^T = (A * B^T)^T = B * A^T
        // Bseg is 1xK, a_pack is DxK.
        // matrix_kernel_matmul_i8_i32_abt computes X * Y^T.
        // Let X=Bseg, Y=a_pack. Result is 1xD.
        // Layout of 1xD in memory is identical to Dx1, so c_pack is correct.
        (void)matrix_kernel_matmul_i8_i32_abt(Bseg, a_pack, c_pack, 1, d, count);
        debug_delay_cycles(2000);
        /* Ensure matrix unit writes are visible to subsequent CPU loads. */
        //asm volatile("fence rw, rw" ::: "memory");

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
 //(ptrdiff_t)sizeof(float), /* ws stride bytes (contiguous) */