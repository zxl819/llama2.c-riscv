#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "matrix_kernel_1230.h"

// Define the global buffer (declared extern in header)
float matrix_kernel_ws_pack_buf[MK_MMAX] __attribute__((aligned(64)));

// The function implementation - Reverted to NOT take stride
void matrix_kernel_rvv_accum_i32_to_f32_ws_xs(float *dst,
                                              const int32_t *src_i32,
                                              const float *ws,
                                              float xs,
                                              int n) {
    debug_delay_cycles(2000);
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t vsrc = __riscv_vle32_v_i32m1(src_i32, vl);
        vfloat32m1_t vfsrc = __riscv_vfcvt_f_x_v_f32m1(vsrc, vl);
        vfloat32m1_t vws = __riscv_vle32_v_f32m1(ws, vl);
        vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vfsrc, vws, vl);

        /* Hardware lacks float scalar->vector path; broadcast xs by
           constructing a u32 vector with xs bitpattern via vmv and
           reinterpret as float vector. */
        union { float f; uint32_t u; } xu; xu.f = xs;
        vuint32m1_t vxs_u = __riscv_vmv_v_x_u32m1(xu.u, vl);
        vfloat32m1_t vxs = __riscv_vreinterpret_v_u32m1_f32m1(vxs_u);

        vres = __riscv_vfmul_vv_f32m1(vres, vxs, vl);
        
        vfloat32m1_t vdst = __riscv_vle32_v_f32m1(dst, vl);
        vdst = __riscv_vfadd_vv_f32m1(vdst, vres, vl);
        __riscv_vse32_v_f32m1(dst, vdst, vl);
        debug_delay_cycles(20);
        
        dst += vl;
        src_i32 += vl;
        ws += vl;
        avl -= vl;
    }
    debug_delay_cycles(200);
}

void matrix_kernel_rvv_zero_f32(float *dst, int n) {
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        __riscv_vse32_v_f32m1(dst, vzero, vl);
        dst += vl;
        avl -= vl;
    }
    //debug_delay_cycles(20);
}

void matrix_kernel_rvv_zero_i32(int32_t *dst, int n) {
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t vzero_i = __riscv_vmv_v_x_i32m1(0, vl);
        __riscv_vse32_v_i32m1(dst, vzero_i, vl);
        dst += vl;
        avl -= vl;
    }
    debug_delay_cycles(20);
}
