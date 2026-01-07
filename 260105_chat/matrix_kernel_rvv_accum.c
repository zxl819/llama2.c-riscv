#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "matrix_kernel_1230.h"

// Define the global buffer (declared extern in header)
float matrix_kernel_ws_pack_buf[MK_MMAX] __attribute__((aligned(64)));

// The function implementation
void matrix_kernel_rvv_accum_i32_to_f32_ws_xs(float *dst,
                                              const int32_t *src_i32,
                                              const float *ws_base,
                                              ptrdiff_t ws_stride_bytes,
                                              float xs,
                                              int n) {
    /* Avoid the slow/unsupported per-chunk tmp-gather path inside the vector loop.
       If ws is strided, materialize it once into a contiguous buffer, then
       always use contiguous access. */
    const float *ws_ptr = ws_base;
    if (ws_stride_bytes != (ptrdiff_t)sizeof(float)) {
        for (int r = 0; r < n; r++) {
            matrix_kernel_ws_pack_buf[r] = *(const float *)((const uint8_t *)ws_base + (size_t)r * (size_t)ws_stride_bytes);
        }
        ws_ptr = matrix_kernel_ws_pack_buf;
    }

    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t vsrc = __riscv_vle32_v_i32m1(src_i32, vl);

// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//         __attribute__((aligned(64)))static float vsrc_buf2[16];
//         /* reinterpret i32 -> u32 -> f32 so we can store with the f32 store intrinsic */
//         vuint32m1_t vsrc_u = __riscv_vreinterpret_v_i32m1_u32m1(vsrc);
//         vfloat32m1_t vsrc_u_f = __riscv_vreinterpret_v_u32m1_f32m1(vsrc_u);
//         __riscv_vse32_v_f32m1(vsrc_buf2, vsrc_u_f, vl);
//         debug_delay_cycles(1000);
//         matrix_kernel_debug_print_prefix();
//         print_uart("   vsrc vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//         matrix_kernel_debug_print_prefix();
//         matrix_kernel_debug_print_arr_f32("   vsrc=", vsrc_buf2, (int)vl);
// #endif

        vfloat32m1_t vfsrc = __riscv_vfcvt_f_x_v_f32m1(vsrc, vl);

// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//         __attribute__((aligned(64)))static float lane_buf[16];
//         __riscv_vse32_v_f32m1(lane_buf, vfsrc, vl);
//         debug_delay_cycles(1000);
//         matrix_kernel_debug_print_prefix();
//         print_uart("   vfsrc vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//         matrix_kernel_debug_print_prefix();
//         matrix_kernel_debug_print_arr_f32("   vfsrc=", lane_buf, (int)vl);
// #endif

        vfloat32m1_t vws = __riscv_vle32_v_f32m1(ws_ptr, vl);

    // #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    //     __attribute__((aligned(64)))static float ws_buf[16];
    //     __riscv_vse32_v_f32m1(ws_buf, vws, vl);
    //     debug_delay_cycles(1000);
    //     matrix_kernel_debug_print_prefix();
    //     print_uart("   ws_buf vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
    //     matrix_kernel_debug_print_prefix();
    //     matrix_kernel_debug_print_arr_f32("   ws_buf=", ws_buf, (int)vl);
    // #endif
        //debug_delay_cycles(20);
        vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vfsrc, vws, vl);

// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//             __attribute__((aligned(64)))static float vres_buf[16];
//             __riscv_vse32_v_f32m1(vres_buf , vres, vl);
//             debug_delay_cycles(1000);
//             matrix_kernel_debug_print_prefix();
//             print_uart("   vres vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//             matrix_kernel_debug_print_prefix();
//             matrix_kernel_debug_print_arr_f32("   vres=", vres_buf, (int)vl);
// #endif

        /* Hardware lacks float scalar->vector path; broadcast xs by
           constructing a u32 vector with xs bitpattern via vmv and
           reinterpret as float vector. */
        union { float f; uint32_t u; } xu; xu.f = xs;
        vuint32m1_t vxs_u = __riscv_vmv_v_x_u32m1(xu.u, vl);

// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//             __attribute__((aligned(64))) static float vxs_u_buf[16];
//             /* reinterpret unsigned u32 vector as float vector then store */
//             vfloat32m1_t vxs_u_f = __riscv_vreinterpret_v_u32m1_f32m1(vxs_u);
//             __riscv_vse32_v_f32m1(vxs_u_buf, vxs_u_f, vl);
//             debug_delay_cycles(1000);
//             matrix_kernel_debug_print_prefix();
//             print_uart("   vxs vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//             matrix_kernel_debug_print_prefix();
//             matrix_kernel_debug_print_arr_f32("   vxs=", vxs_u_buf, (int)vl);
// #endif
        vfloat32m1_t vxs = __riscv_vreinterpret_v_u32m1_f32m1(vxs_u);
// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//            __attribute__((aligned(64)))static float vxs_buf[16];
//             __riscv_vse32_v_f32m1(vxs_buf, vxs, vl);
//             debug_delay_cycles(1000); 
//             matrix_kernel_debug_print_prefix();
//             print_uart("   vxs vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//             matrix_kernel_debug_print_prefix();
//             matrix_kernel_debug_print_arr_f32("   vxs=", vxs_buf, (int)vl);
// #endif
        vres = __riscv_vfmul_vv_f32m1(vres, vxs, vl);
        
        vfloat32m1_t vdst = __riscv_vle32_v_f32m1(dst, vl);

        vdst = __riscv_vfadd_vv_f32m1(vdst, vres, vl);
// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//            __attribute__((aligned(64))) static float vdst_buf[16];
//             __riscv_vse32_v_f32m1(vdst_buf, vdst, vl);
//             debug_delay_cycles(1000);
//             matrix_kernel_debug_print_prefix();
//             print_uart("   vdst vl="); matrix_kernel_debug_print_u64((uint64_t)vl); print_uart(" vals=\r\n");
//             matrix_kernel_debug_print_prefix();
//             matrix_kernel_debug_print_arr_f32("   vdst=", vdst_buf, (int)vl);
// #endif
        __riscv_vse32_v_f32m1(dst, vdst, vl);
        //debug_delay_cycles(50);
        
        dst += vl;
        src_i32 += vl;
        ws_ptr += vl;
        avl -= vl;
    }
    debug_delay_cycles(50);
}

void matrix_kernel_rvv_zero_f32(float *dst, int n) {
// #if MATRIX_KERNEL_DEBUG_PRINT
// //print_uart("=== matrix_kernel_rvv_zero_f32 entered ===\r\n");
// debug_delay_cycles(50);
// #endif
    float *dst_orig = dst; int n_orig = n;
    // print_uart("rvv_zero_f32 dst=");
    // print_uart_hex((uint64_t)(uintptr_t)dst);
    // print_uart("\r\n");
// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//     static uint64_t s_zero_f32_calls = 0;
//     const uint64_t dbg_id = ++s_zero_f32_calls;
//     if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
//         matrix_kernel_debug_print_prefix();
//         print_uart("rvv_zero_f32 dst=");
//         matrix_kernel_debug_print_ptr(dst);
//         print_uart(" n=");
//         matrix_kernel_debug_print_u64((uint64_t)n);
//         print_uart("\r\n");
//     }
// #endif
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        __riscv_vse32_v_f32m1(dst, vzero, vl);
        dst += vl;
        avl -= vl;
    }
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    // matrix_kernel_debug_print_prefix();
    // print_uart("rvv_zero_f32 post dst=");
    // matrix_kernel_debug_print_ptr(dst_orig);
    // print_uart(" n=");
    // matrix_kernel_debug_print_u64((uint64_t)n_orig);
    // print_uart(" vals=\r\n");
    // matrix_kernel_debug_print_arr_f32("  zeroed_f32", dst_orig, n_orig);
#endif
    //debug_delay_cycles(500);
}

void matrix_kernel_rvv_zero_i32(int32_t *dst, int n) {
// #if MATRIX_KERNEL_DEBUG_PRINT
// //print_uart("=== matrix_kernel_rvv_zero_i32 entered ===\r\n");
// debug_delay_cycles(50);
// #endif
    int32_t *dst_orig = dst; int n_orig = n;
// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//     static uint64_t s_zero_i32_calls = 0;
//     const uint64_t dbg_id = ++s_zero_i32_calls;
//     if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
//         matrix_kernel_debug_print_prefix();
//         print_uart("rvv_zero_i32 dst=");
//         matrix_kernel_debug_print_ptr(dst);
//         print_uart(" n=");
//         matrix_kernel_debug_print_u64((uint64_t)n);
//         print_uart("\r\n");
//     }
// #endif
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        // Construct an i32 zero vector directly and store with i32 stores to avoid float-int store issues.
        vint32m1_t vzero_i = __riscv_vmv_v_x_i32m1(0, vl);
        __riscv_vse32_v_i32m1(dst, vzero_i, vl);
        // small delay for stability on some targets
        //debug_delay_cycles(20);
        dst += vl;
        avl -= vl;
    }
// #if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
//     matrix_kernel_debug_print_prefix();
//     print_uart("rvv_zero_i32 post dst=");
//     matrix_kernel_debug_print_ptr(dst_orig);
//     print_uart(" n=");
//     matrix_kernel_debug_print_u64((uint64_t)n_orig);
//     print_uart(" vals=\r\n");
//     matrix_kernel_debug_print_arr_i32("  zeroed_i32", dst_orig, n_orig);
// #endif
    debug_delay_cycles(20);
}
