// Lightweight wrapper: qmatmul that avoids MK_MMAX blocking and relies
// on matrix_kernel_matmul_i8_i32_abt's internal msettile for m-tiling.
// This header expects matrix_kernel.h to be available and uses its
// helper functions (ml* / msettile / rvv zero helpers).

#ifndef MATRIX_KERNEL_NOBLK_H
#define MATRIX_KERNEL_NOBLK_H

#include "matrix_kernel_1230.h"

// Top-level API: same signature as matrix_kernel_qmatmul_f32 but this
// implementation does NOT split D by MK_MMAX. Instead it packs the full
// D (d) rows per group and calls the matrix matmul routine which will
// internally tile m via msettile* instructions.
static BARE_NO_AUTOVEC inline void matrix_kernel_qmatmul_f32_noblk(float *xout,
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
        int8_t a_pack[d * count];
        for (int r = 0; r < d; r++) {
            const int8_t *src = wq + (ptrdiff_t)r * (ptrdiff_t)w_row_stride + (ptrdiff_t)base;
            int8_t *dst = a_pack + (size_t)r * (size_t)count;
            for (int c = 0; c < count; c++) dst[c] = src[c];
        }

        // B segment (single row) for n=1 output
        const int8_t *Bseg = xq + base;

        // Allocate contiguous C buffer [d] and clear it.
        int32_t c_pack[d];
        matrix_kernel_rvv_zero_i32(c_pack, d);

        // Call the non-strided abt path with n=1; m tiling is internal.
        (void)matrix_kernel_matmul_i8_i32_abt(a_pack, Bseg, c_pack, d, 1, count);
        /* Ensure matrix unit writes are visible to subsequent CPU loads. */
        //asm volatile("fence rw, rw" ::: "memory");

        // Materialize ws for these rows (contiguous)
        // Use a VLA float buffer for ws per-row
        float ws_local[d];
        for (int r = 0; r < d; r++) {
            ws_local[r] = ws[(ptrdiff_t)r * (ptrdiff_t)groups_per_row + (ptrdiff_t)g];
        }

        // RVV-accelerated accumulation: xout[0:d] += float(c_pack[0:d]) * ws_local[0:d] * xs[g]
        matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
            xout,
            c_pack,
            ws_local,
            (ptrdiff_t)sizeof(float), /* ws stride bytes (contiguous) */
            xs[g],
            d
        );
    }
}

#endif // MATRIX_KERNEL_NOBLK_H
