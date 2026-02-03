// ---
// author: zhaoxinlei
// version: 1.0
// ---
// [代码说明]
// 本文件提供了基于 RISC-V Vector (RVV) 指令的辅助函数，用于加速矩阵乘法后的数据处理。
// 核心函数 matrix_kernel_rvv_accum_i32_to_f32_ws_xs 实现了以下操作：
//   dst[i] += (float)src_i32[i] * ws[i * stride] * xs
// 该实现特别针对 FPGA 上的 AME (Matrix) 与 CPU 缓存一致性做了优化。

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "matrix_kernel_1230.h"

// Define the global buffer in Uncached region
//float matrix_kernel_ws_pack_buf[MK_MMAX] __attribute__((aligned(64), section(".uncached_buffer")));

// Local static buffer to avoid stack allocation and overflow
#define MAX_TMP_WS_ELEMS 64 // 256 -> 64 (Safe for vl=16)
//__attribute__((section(".uncached_buffer"), aligned(64))) 
__attribute__((aligned(64)))
static float matrix_kernel_tmp_ws_buf[MAX_TMP_WS_ELEMS];

// The function implementation - Supports stride for ws
// Note: Use volatile to force memory access and bypass any potential CPU optimization
void matrix_kernel_rvv_accum_i32_to_f32_ws_xs(volatile float *dst,
                                              const volatile int32_t *src_i32,
                                              const float *ws_base,
                                              int ws_stride_in_floats,
                                              float xs,
                                              int n) {
    debug_delay_cycles(200);
static int s_accum_call_count = 0;
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    // [New Debug] Print first 4 intermediate values from Matrix Unit
    static int s_accum_call_count = 0;
    if (s_accum_call_count < 4) { // Just for the first couple of groups
        print_uart("    [rvv_accum] src_i32 addr=0x");
        print_uart_hex((unsigned long long)(uintptr_t)src_i32);
        print_uart(" vals[0..3]: ");
        for(int k=0; k<4; k++) {
             print_dec32(src_i32[k]); print_uart(" ");
        }
        print_uart("\r\n");
    }
    s_accum_call_count++;
#endif
s_accum_call_count++;
    size_t avl = (size_t)n;
    int offset = 0;
    
    while (avl > 0) {
        size_t request_avl = (ws_stride_in_floats != 1 && avl > MAX_TMP_WS_ELEMS) ? MAX_TMP_WS_ELEMS : avl;
        
        // if (s_accum_call_count < 20 && offset >= 256) {
        //      print_uart("LoopTop: off="); print_dec32((int)offset); 
        //      print_uart(" avl="); print_dec32((int)avl);
        //      print_uart("\r\n");
        // }

        // Wait for previous operations/Matrix unit to settle before VSETVL/Load
        debug_delay_cycles(500); 
        size_t vl = __riscv_vsetvl_e32m1(request_avl);

        // Load from AME output (src_i32 is already in .uncached_buffer)
        // Ensure AME write is visible (using delay instead of fence)
        // if (s_accum_call_count < 20 && offset >= 256) { print_uart("  > Load src_i32... "); }
        // debug_delay_cycles(200);
        vint32m1_t vsrc = __riscv_vle32_v_i32m1((const int32_t *)src_i32, vl);
        // if (s_accum_call_count < 20 && offset >= 256) { print_uart("Done\r\n"); }
        
        vfloat32m1_t vfsrc = __riscv_vfcvt_f_x_v_f32m1(vsrc, vl);
        
        vfloat32m1_t vws;
        if (ws_stride_in_floats == 1) {
             vws = __riscv_vle32_v_f32m1((const float *)(ws_base + offset), vl);
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
             if (s_accum_call_count <= 2) {
                 print_uart("    [rvv_accum] ws_chunk[0..3] (direct): ");
                 for(int k=0; k<4; k++) {
                      print_float_fixed3(ws_base[offset + k]); print_uart(" ");
                 }
                 print_uart("\r\n");
             }
#endif
        } else {
             // Use uint32_t pointer to force 'lw' instruction (integer load) instead of 'flw'.
             // This bypasses potential FPU data path hazards or strict alignment strictness for floats on some cores.
             const volatile uint32_t *curr_ws_ptr_u = (const volatile uint32_t *)(const void *)ws_base;
             
             // Calculate start address for this block. Pointer arithmetic on uint32_t* matches float* (both 4 bytes).
             const volatile uint32_t *loader_ptr = curr_ws_ptr_u + ((size_t)offset * (size_t)ws_stride_in_floats);
             uint32_t *tmp_u = (uint32_t *)(void *)matrix_kernel_tmp_ws_buf;

            //  if (s_accum_call_count < 20) { 
            //      print_uart("[DEBUG] offset: "); print_dec32((int)offset); 
            //      print_uart(" ptr: 0x"); print_uart_hex((uintptr_t)loader_ptr);
            //      print_uart(" vl: "); print_dec32((int)vl);
            //      print_uart("\r\n");
            //  }

             // Delay before scalar loop to ensure pointers are stable
             //debug_delay_cycles(1000);

             // Scalar loop to gather strided elements into contiguous buffer
             // Corresponds to ASM address ~80004318 (lw)
             for(size_t i=0; i<vl; ++i) {
                  //if (s_accum_call_count < 20 && offset >= 256) { print_uart("."); }
                  tmp_u[i] = *loader_ptr;
                  loader_ptr += ws_stride_in_floats;
             }
         //    if (s_accum_call_count < 20 && offset >= 256) { print_uart(" OK\r\n"); }
             
             // Wait for scalar stores (tmp_u) to complete before vector load reads them
             // Replaces 'fence w, r'
             debug_delay_cycles(200);
             
          //   if (s_accum_call_count < 20 && offset >= 256) { print_uart("  > VLE... "); }
             vws = __riscv_vle32_v_f32m1(matrix_kernel_tmp_ws_buf, vl);
          //   if (s_accum_call_count < 20 && offset >= 256) { print_uart("Done\r\n"); }
        }

      //  if (s_accum_call_count < 20 && offset >= 256) { print_uart("  > VMULs... "); }
        vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vfsrc, vws, vl);

        union { float f; uint32_t u; } xu; xu.f = xs;
        vuint32m1_t vxs_u = __riscv_vmv_v_x_u32m1(xu.u, vl);
        vfloat32m1_t vxs = __riscv_vreinterpret_v_u32m1_f32m1(vxs_u);

        vres = __riscv_vfmul_vv_f32m1(vres, vxs, vl);
        //if (s_accum_call_count < 20 && offset >= 256) { print_uart("Done\r\n"); }
        
        // Load current accumulation result from dst
       // if (s_accum_call_count < 20 && offset >= 256) { print_uart("  > VADD/Store... "); }
        vfloat32m1_t vdst = __riscv_vle32_v_f32m1((const float *)dst, vl);
        vdst = __riscv_vfadd_vv_f32m1(vdst, vres, vl);
        __riscv_vse32_v_f32m1((float *)dst, vdst, vl);
       // if (s_accum_call_count < 20 && offset >= 256) { print_uart("Done\r\n"); }
        
        // 确保写回内存
        //__asm__ volatile("fence w, w" ::: "memory");
        debug_delay_cycles(500);
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
        if (s_accum_call_count <= 4) {
            print_uart("    [rvv_accum] dst_chunk[0..15]: ");
            for(int k=0; k<16; k++) {
                 print_float_fixed3(dst[k]); print_uart(" ");
            }
            print_uart("\r\n");
        }
#endif
        dst += vl;
        src_i32 += vl;
        offset += vl; 
        avl -= vl;
    }
    debug_delay_cycles(1000);
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
    debug_delay_cycles(200);
}
