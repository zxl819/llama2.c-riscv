/*
---
title: 16x16 矩阵 load/store 行为测试 (AME Inline)
date: 2026-03-20
description: 使用 RISC-V 矩阵扩展 (AME) 指令对 16x16 int32 累加器执行单次加载与单次写回，并打印 load 前和 store 后的矩阵值。
author: zhaoxinlei
version: 1.0
---
============================================================================================================================================================
代码说明
============================================================================================================================================================
该测试例用于验证矩阵寄存器与内存之间的 load/store 路径。
当前版本默认只做一次矩阵加载与一次矩阵写回，不执行 mqma 乘加计算。
主要步骤：
1. 静态分配 64 字节对齐的 A (int8), B (int8), C (int32) 矩阵。
2. 配置矩阵形状：通过 msettilem, msettilen, msettilek 设置 16x16x16 的计算维度。
3. 在 mlce32.m 之前打印 C（load 前）。
4. 执行一次 mlce32.m 将 C 载入累加器。
5. 可选加载 A/B 输入（mlae8.m, mlbe8.m）用于保持数据通路。
6. 执行一次 msce32.m 将累加器写回 C，并打印写回后的 C（store 后）。
7. 主程序中可继续与 Golden 数据 D_ref_init 对比，用于观测是否存在差异。
============================================================================================================================================================
*/

#include <stdint.h>
//#include <riscv_matrix.h>
#include <stddef.h>
//#include "matrix/matrix_intrinsic.h"

// If you want to link this file into a larger program (e.g. RVV llama2 model),
// define MATMUL_MATRIX_KERNEL_ONLY=1 for this translation unit.
// That disables the standalone demo (main/UART/data globals) and only exposes
// a callable Matrix-kernel function.
#ifndef MATMUL_MATRIX_KERNEL_ONLY
//#include "uart.h"

#include "uart_helper.c"
// Standalone build historically included uart.c directly.
// Keep it ONLY for the standalone demo to avoid multiple-definition issues when
// linking into other programs that also compile uart.c.
//#include "uart.c"
#endif

//A*B

#ifndef MATMUL_MATRIX_KERNEL_ONLY
//用于UART输出调试信息
#define CLOCK_FREQUENCY 24000000 //50MHz
#define UART_BITRATE    115200
#endif

#define DIM_M 16
#define DIM_K 16
#define DIM_N 16


// 简单的空转延时函数，使用 asm volatile 防止被优化掉
static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}
/* 其中.matA,B,C由link.ld中定义
    . = 0x80040000; _matA_start = .; .matA : { *(.matA) } _matA_end = .;
    . = 0x80050000; _matB_start = .; .matB : { *(.matB) } _matB_end = .;
    . = 0x80060000; _matC_start = .; .matC : { *(.matC) } _matC_end = .;*/

#ifndef MATMUL_MATRIX_KERNEL_ONLY
// 8x8x8 int8_t matmul，result in int32_t
// NOTE: When building the standalone demo, matrices A/B/C (and golden D_ref)
// are provided by data.h. Do not redefine them here.
#endif

// ---------------------------------------------------------------------------
// Matrix inline-asm helpers (match mnemonics emitted by Matrix clang)
static inline int asm_msettilem(int rem) {
  // Encoding from Matrix clang14 for: msettilem a0, a0
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04055577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}

static inline int asm_msettilen(int rem) {
  // Encoding from Matrix clang14 for: msettilen a0, a0
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04054577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}

static inline int asm_msettilek(int rem) {
  // Encoding from Matrix clang14 for: msettilek a0, a0
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04056577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}

static inline void asm_mlce32_acc0(const int32_t *base, int stride_bytes) {
  // Encoding from Matrix clang14 for: mlce32.m acc0, (a0), a1
  register const int32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x00b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

static inline void asm_msce32_acc0(int32_t *base, int stride_bytes) {
  // Encoding from Matrix clang14 for: msce32.m acc0, (a0), a1
  register int32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x02b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

static inline void asm_mlae8_tr0(const int8_t *base, int stride_bytes) {
  // Encoding from Matrix clang14 for: mlae8.m tr0, (a0), a1
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

static inline void asm_mlbe8_tr1(const int8_t *base, int stride_bytes) {
  // Encoding from Matrix clang14 for: mlbe8.m tr1, (a0), a1
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b500f7\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

static inline void asm_mqma_b_acc0_tr0_tr1(void) {
  // Encoding from Matrix clang14 for: mqma.b.mm acc0, tr0, tr1
  asm volatile(".word 0x28180877" ::: "memory");
}

#ifndef MATMUL_MATRIX_KERNEL_ONLY
static void print_mat_i32(const int32_t* m, int rows, int cols);
#endif

static inline int matmul_batch1(const int8_t *A, const int8_t *B, int32_t *C,
                                 int m, int n, int k) {
  // lda, ldb, ldc in bytes for msetilem/len/k                                
  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = n * (int)sizeof(int8_t);
  // C buffer is a 4x4 (see data.h). We only compute/update the top-left m x n.
  // Keep the stride configurable via macro for reuse.
  #ifndef MATMUL_MATRIX_LDC_ELEMS
  #define MATMUL_MATRIX_LDC_ELEMS 16
  #endif
  const int ldc_bytes = MATMUL_MATRIX_LDC_ELEMS * (int)sizeof(int32_t);

  // 固定单 tile：只执行 1 次矩阵 load 和 1 次矩阵 store。
  // 对于 16x16 测试，m/n/k 都是 16，这里会完整覆盖矩阵。
  const int tile_m = asm_msettilem(m);
  const int tile_n = asm_msettilen(n);
  const int tile_k = asm_msettilek(k);
  int32_t *cptr = C;

  (void)tile_m;
  (void)tile_n;
  (void)tile_k;

// #ifndef MATMUL_MATRIX_KERNEL_ONLY
  print_uart("C before mlce32(load):\r\n");
  print_mat_i32((const int32_t *)cptr, m, n);
// #endif

  asm_mlce32_acc0((const int32_t *)cptr, ldc_bytes);

  // 保留 A/B 的单次加载，用于形成完整的数据通路。
  asm_mlae8_tr0(A, lda_bytes);
  asm_mlbe8_tr1(B, ldb_bytes);
  // 这里保持只验证 load/store，不执行乘加。
  // asm_mqma_b_acc0_tr0_tr1();

  asm_msce32_acc0(cptr, ldc_bytes);
  debug_delay_cycles(1000); // 确保 store 完成

// #ifndef MATMUL_MATRIX_KERNEL_ONLY
  print_uart("C after msce32(store):\r\n");
  print_mat_i32((const int32_t *)cptr, m, n);
// #endif

  return 0;
}

// Public entry for linking into other programs.
// A: [m x k] int8
// B: [k x n] int8 (assumes the addressing pattern used by mlb_m in this file)
// C: [m x n] int32
int matmul_matrix_i8_i32(const int8_t *A, const int8_t *B, int32_t *C,
                         int m, int n, int k) {
  return matmul_batch1(A, B, C, m, n, k);
}



#ifndef MATMUL_MATRIX_KERNEL_ONLY
// 打印 int32 矩阵，rows x cols，每行结尾换行
static void print_mat_i32(const int32_t* m, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            print_dec32(m[r * cols + c]);
      if (c + 1 < cols) write_serial((uint8_t)' ');
        }
        print_uart("\r\n");
    }
}

static void print_mat_i8(const int8_t* m, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
      print_dec32((int32_t)m[r * cols + c]);
      if (c + 1 < cols) write_serial((uint8_t)' ');
        }
        print_uart("\r\n");
    }
}

/* 初始化数据 (全局静态变量，避免栈拷贝问题) */
__attribute__((aligned(64))) static const int8_t A_init[DIM_M*DIM_K] = {
  100, -7, 61, 3, 40, 65, 22, 115, -104, -120, 36, -92, -17, 82, 48, 114,
  -14, -65, -81, -49, -16, 94, -97, -20, -19, -59, -47, 83, -70, -63, -58, -24,
  122, 36, -84, 12, -126, -107, -99, -121, -78, 93, -68, 64, 28, 86, 66, 9,
  25, 81, -52, -44, 85, -13, -10, 73, -103, -97, -83, -51, -116, -97, -16, -12,
  2, 122, 14, -94, 53, -30, -23, -25, -91, 103, 36, -76, 125, 0, 119, -61,
  -106, -123, 102, 64, 23, -113, 94, -57, 124, -1, -7, -4, 38, -99, 23, 123,
  -75, 63, -86, 118, -93, -105, 85, 57, 33, -53, 87, 10, -57, 108, -41, -58,
  114, 57, -3, -87, -48, -46, -120, 120, -39, -21, -110, 4, 124, -54, -34, -99,
  15, -20, -122, 31, -99, -12, 52, 70, 80, -36, 72, 28, 12, 69, -123, 106,
  -30, -19, 66, -118, -19, 55, 32, 7, 37, 95, -119, -11, -14, -34, -92, -113,
  -68, -11, -107, 36, -25, 69, -91, 90, -125, -73, 123, 23, 125, 77, 63, -62,
  -95, -40, -7, 87, -123, 20, -46, 2, -22, 44, 29, 2, 120, 123, 22, 64,
  -1, -115, -73, -91, 100, 11, -114, 81, -126, -111, 100, 46, -74, 66, -81, 73,
  25, 95, 67, -79, 96, 14, 60, 77, 57, -37, 26, -80, 72, -6, 33, -108,
  97, -72, -89, 90, 26, 42, -109, 92, -115, 87, 60, 96, 51, -49, -109, -8,
  -87, 30, -119, -58, -78, 107, 69, -127, -32, 86, 118, 37, -103, -64, 123, 56
};

__attribute__((aligned(64))) static const int8_t B_init[DIM_K*DIM_N] = {
  67, -23, 93, 85, 82, 127, -91, -56, -105, -8, 9, -73, -32, 49, -72, 35,
  46, 88, -110, 78, 63, 122, -94, 118, -28, 101, 75, -90, -112, -118, -59, -5,
  -15, -40, -103, 101, 5, 76, -84, -20, -96, 19, 27, 22, 30, 100, -123, -122,
  4, -3, 93, 44, -83, -13, -28, 107, 9, 115, 83, 83, -87, -10, -12, 98,
  49, -110, -116, 41, -74, -60, 3, -66, -10, 45, 7, 68, -91, 99, -48, -74,
  35, 95, -19, 84, 58, -48, -56, -112, -2, 69, -85, 83, -37, -11, -91, -86,
  87, -91, 34, -32, -22, 65, 64, -47, -80, -120, -10, 48, -10, 62, 98, -83,
  -76, 15, -117, -27, 119, 0, 56, -127, -87, 34, 90, -61, -128, 13, -121, -21,
  -113, 28, 39, -101, -43, -35, 69, 34, -126, 68, -4, -31, -45, -122, 93, 57,
  94, 1, 67, 39, -45, -88, 96, -18, -24, 98, 55, 94, -41, -48, 27, 33,
  -18, -107, -93, 79, -20, -60, 91, -41, -12, 119, 105, 11, 9, 95, 3, -78,
  115, 66, -101, 127, 82, -109, -34, -66, 113, -91, -112, -63, -25, -49, 21, -110,
  110, 101, -30, -63, -1, 107, -28, 67, 96, -84, -89, 50, 22, 70, 58, -96,
  112, -95, -86, -32, -71, -111, 1, -21, -120, -6, -24, 42, 65, 18, 66, -12,
  -28, -72, -112, 22, -2, -12, 17, 86, 41, 50, -84, 57, 111, 18, -99, -35,
  84, -56, -103, -14, 90, 18, 15, -34, 33, -91, 18, -100, 59, -117, 35, -76
};

__attribute__((aligned(64))) static int32_t C_init[16*16] = {
  -54, 24, 511, 901, -931, -712, 646, 898, -502, -377, 738, -153, -454, 656, -486, -182,
  288, 99, -829, -945, 732, 507, 676, 76, 635, -341, -95, 577, -753, -394, -752, -93,
  954, -732, -233, -194, 808, -593, 5, -476, -961, 501, -876, -439, -4, -30, -767, 962,
  498, 924, -816, 450, -414, 82, 850, -446, 452, -679, -355, 940, -158, 32, -414, -769,
  -151, 247, -89, 554, -274, 226, 546, 835, -145, -921, 437, 57, 745, -81, -264, -876,
  -86, 283, 540, 706, -571, 186, 609, -480, -311, 680, 163, 19, 348, 22, 962, 506,
  -892, -705, 90, 640, -862, 367, 519, 574, 747, -617, 111, 605, -284, -618, -41, -837,
  -560, 711, 335, 723, 680, 753, -379, -56, 235, -452, 839, -986, 677, 292, -494, 440,
  -174, 671, 997, -436, -60, -570, 385, 279, 689, 610, 954, 928, 790, -699, -916, -36,
  -310, 790, 594, -155, 151, 179, 782, -951, -21, 347, -94, 839, 902, 654, -71, 771,
  -854, 321, -464, -509, 359, 537, 777, -577, 744, 663, -371, -875, 544, 651, -79, -671,
  -709, -250, 515, -367, -937, 383, 489, -643, 121, -208, 2, -989, 269, -475, 109, -158,
  220, -789, -272, 266, 535, -239, -951, 451, 9, 308, -682, -138, 767, 735, -368, 264,
  -832, 621, -464, -317, 932, 87, 749, -608, 516, 993, -846, -514, -707, -487, -378, -854,
  796, -485, 843, 527, -655, 396, 547, -743, -741, -248, -863, -158, -40, 330, 144, -88,
  -560, 173, -90, 680, 398, 453, 144, -270, -437, -103, 144, -265, -708, -781, -913, -594
};

__attribute__((aligned(64))) static const int32_t D_ref_init[16*16] = {
  39844, 10082, -11167, 6522, -28037, -46909, 29749, -13778, 886, -8371, -24537, -21231, 11322, 5374, -27829, 3230,
  8194, 18435, 23909, -8717, 18699, 10973, 11537, 12186, -7717, -22137, -18285, -789, 12564, 1796, 10477, 9508,
  -5585, -24246, 17386, 1031, 47468, 35528, 8876, -15592, -10670, 24670, 20213, 20447, 5710, 61057, 963, -18404,
  8133, 41707, -9768, -18386, -19004, 8064, 1344, 23504, 24080, -16212, -41889, 31068, 12779, -2524, -13622, 21442,
  -5582, 2869, -8970, -19253, -31559, 39, -3924, 1784, 32066, -1653, 824, -7781, 5611, 13383, -16735, 9117,
  -23461, -47828, -51016, 13332, -17392, -16935, -19551, -15416, 17229, 22921, 7039, -1701, -53549, -12928, 2024, 211,
  -32172, -4183, 22349, 11088, 33735, 3127, -9065, 17975, -14959, 5666, 42288, 16807, -6855, -3318, 11215, -20930,
  -12135, 6665, 6854, -13423, -13916, 11536, -991, -50683, -6902, -14218, -35957, 21672, 45561, 30207, 26002, 15535,
  -19659, 5355, 13747, 19119, 25667, -18507, -18344, 3123, -26849, 1795, 21831, -11042, -2344, 2020, 29765, -11564,
  -7114, -3198, 425, -6501, -973, 14505, -3670, -23, 11063, -6875, -9849, -12549, 15051, -16526, 19206, -6255,
  4110, 13609, 52321, -11638, 1824, -26783, 19531, -6788, -7361, -31201, 18660, -27921, 10962, 22130, 14206, 7032,
  2034, -23850, 26938, 15774, 8823, -17561, -6634, -24889, -6967, -8426, 26914, -39101, -17714, 9390, 17835, -33676,
  29358, 18484, 31099, 5353, 15972, -23856, 12712, 37009, -8143, -24153, -6138, -35846, -11425, 16076, 754, 2819,
  -10317, 2855, -21111, -24746, -40381, -12840, 4990, -11800, 166, -12753, -19026, 14058, 35198, -22429, -7738, 25334,
  29003, 56723, 58601, 25501, 23534, 19854, -10308, 2793, -22980, 10023, 18941, -12111, -12708, 25464, 35980, 3635,
  -20753, 20698, -11845, 12265, 12723, -14108, 4015, 43356, 46838, 16146, 24020, -40515, -22874, -3243, -30999, -5252
};

int main()
{
  print_uart("Matrix Inline-asm 16x16 int8_t matmul demo\r\n");
    /* Definitions moved from data.h */
    #define M DIM_M
    #undef M
    #undef K
    #undef N
      // 初始化 UART
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    // 打印计算结果 C（int32）
    // print_uart("C original:\r\n");
    // print_mat_i32((int32_t*)C,4, 4);
    
    // 计算
    print_uart("Matrix B (KxN):\r\n");
    print_uart("B address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)B_init);
    print_uart("\r\n");
    print_mat_i8((const int8_t*)B_init, DIM_K, DIM_N);

    debug_delay_cycles(1000000);
    matmul_matrix_i8_i32(A_init, B_init, C_init, DIM_M, DIM_N, DIM_K);
    debug_delay_cycles(1000000);

    print_uart("Matrix A:\r\n");
    print_uart("A address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)A_init);
    print_uart("\r\n");
    print_mat_i8((const int8_t*)A_init, DIM_M, DIM_K);

    print_uart("Matrix B (KxN):\r\n");
    print_uart("B address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)B_init);
    print_uart("\r\n");
    print_mat_i8((const int8_t*)B_init, DIM_K, DIM_N);

    // 打印计算结果 C（int32）
    print_uart("C result:\r\n");
    print_uart("C address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)C_init);
    print_uart("\r\n");
    print_mat_i32((int32_t*)C_init, 16, 16);

    print_uart("Golden D_ref (int32):\r\n");
    print_uart("D_ref address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)D_ref_init);
    print_uart("\r\n");
    print_mat_i32((const int32_t*)D_ref_init, 16, 16);

    int mismatch = 0;
    for (int idx = 0; idx < 16 * 16; ++idx) {
      if (C_init[idx] != D_ref_init[idx]) {
        mismatch++;
      }
    }
    if (mismatch == 0) {
      print_uart("Compare: PASS\r\n");
    } else {
      print_uart("Compare: FAIL mismatches=");
      print_dec32(mismatch);
      print_uart("\r\n");
    }
    print_uart("test end\r\n");

    return 0;
}

  #endif
