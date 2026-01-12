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
extern void matrix_kernel_qmatmul_f32_noblk(float *xout,
                                                                  const int8_t *xq,
                                                                  const float *xs,
                                                                  const int8_t *wq,
                                                                  const float *ws,
                                                                  int n,
                                                                  int d,
                                                                  int gs,
                                                                  int w_row_stride);
extern void matrix_kernel_qmatmul_f32_noblk_CT(float *xout,
                                                                  const int8_t *xq,
                                                                  const float *xs,
                                                                  const int8_t *wq,
                                                                  const float *ws,
                                                                  int n,
                                                                  int d,
                                                                  int gs,
                                                                  int w_row_stride);

extern void matrix_kernel_qmatmul_f32_noblk_batch(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride,
                                      int batch);

#endif // MATRIX_KERNEL_NOBLK_H
