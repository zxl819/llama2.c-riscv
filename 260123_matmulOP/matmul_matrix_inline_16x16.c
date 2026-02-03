#include <stdint.h>
//#include <riscv_matrix.h>
#include <stddef.h>
//#include "matrix/matrix_intrinsic.h"

// If you want to link this file into a larger program (e.g. RVV llama2 model),
// define MATMUL_MATRIX_KERNEL_ONLY=1 for this translation unit.
// That disables the standalone demo (main/UART/data globals) and only exposes
// a callable Matrix-kernel function.
#ifndef MATMUL_MATRIX_KERNEL_ONLY
#include <string.h>
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
#define CLOCK_FREQUENCY 25000000 //50MHz
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

  // matmul
  int tile_m = 0, tile_n = 0, tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    tile_m = asm_msettilem(m - i);
    //printf("i = %d\n", i);  // 添加调试
    //printf("debug: i=%d remaining=%d tile_m=%d\n", i, m - i, tile_m);
    for (int j = 0; j < n; j += tile_n) {
      tile_n = asm_msettilen(n - j);
      //printf("debug: j=%d remaining=%d tile_n=%d\n", j, n - j, tile_n);
      // SET_MBA0_I32();
      //mint32_t acc1;
      //initialize acc using C

      // C is stored in a larger row-major buffer with stride MATMUL_MATRIX_LDC_ELEMS.
      int32_t *cptr = C + i * MATMUL_MATRIX_LDC_ELEMS + j;
      asm_mlce32_acc0((const int32_t *)cptr, ldc_bytes);
      
      for (int kk = 0; kk < k; kk += tile_k) {
        tile_k = asm_msettilek(k - kk);
        const int8_t *aptr = A + i * k + kk;
        const int8_t *bptr = B + j * n + kk;
        asm_mlae8_tr0(aptr, lda_bytes);
        // ABT addressing pattern as in original code
        asm_mlbe8_tr1(bptr, ldb_bytes);
        asm_mqma_b_acc0_tr0_tr1();
        //printf("debug: kk=%d remaining=%d tile_k=%d\n", kk, k - kk, tile_k);
      }
      asm_msce32_acc0(cptr, ldc_bytes);
    }
    //printf("debug: i=%d remaining=%d tile_m=%d\n", i, m - i, tile_m);
  }
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

int main()
{
  print_uart("Matrix Inline-asm 16x16 int8_t matmul demo\r\n");
    /* Definitions moved from data.h */
    #define M DIM_M
    #define K DIM_K
    #define N DIM_N

    /* 手动在栈上分配并对齐内存，确保变量位于 SRAM (0x100000 - 0x200000) */
    uint8_t stack_buf[262144]; // 256KB 栈缓冲区
    uintptr_t buf_addr = (uintptr_t)stack_buf;
    buf_addr = (buf_addr + 63) & ~63; // Align to 64 bytes

    int8_t *A = (int8_t *)(buf_addr);
    int8_t *B = (int8_t *)(buf_addr + 256); // 16*16 = 256 bytes
    int32_t *C = (int32_t *)(buf_addr + 512); // A + B = 512 bytes
    int32_t *D_ref = (int32_t *)(buf_addr + 1536); // C is 16*16*4 = 1024 bytes

    /* 初始化数据 (存储在 ROM 中，运行时拷贝到栈) */
    const int8_t A_init[M*K] = {
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
    const int8_t B_init[K*N] = {
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
    const int32_t C_init[16*16] = {
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
    const int32_t D_ref_init[16*16] = {
  7620, -36146, -43817, 13330, 26862, 24264, -17884, -31375, -29516, -6039, 13359, -11010, 6948, 32744, -45478, -28858,
  -15190, 31656, 15697, 5274, 19258, -25180, -9927, -23480, 32155, -11920, -25500, -12877, -2397, -17564, -5304, 9229,
  40516, 12654, 30855, 8380, 2882, 3484, -19518, 39902, 24329, -12835, -18856, -14003, 31947, -26729, 16715, 35933,
  -22239, 2966, -7427, 6261, 24221, 25644, -18547, -10601, 3771, -3264, 15113, -26414, -26461, -1088, -38263, 12642,
  22326, -143, -30057, 10579, -6210, 25509, -3604, 28174, 12424, 19173, -816, 19792, 9788, 23118, -21079, -19009,
  -16098, -26394, 6955, -28864, -26381, 2389, 29710, 16915, 7044, -26353, 184, 9547, 22161, -1139, 36923, -11942,
  -13451, -14570, 6644, -18419, -20265, -14837, 24478, 23426, -19270, 12451, 34036, -6653, -14570, -13253, 34216, 28204,
  -9024, 53683, 7034, -5012, 38647, 47641, -32393, 2939, 6819, -13470, -7251, -32220, -18837, 3395, -31353, 13615,
  6694, 3346, 13079, -27694, 14454, -10879, 24799, -16879, -13160, -18712, 16615, -28929, -8660, -30747, 44697, 10028,
  -8048, 29936, 28823, -9387, -603, 1211, 678, -21899, -18348, -8581, -12729, 11874, -11039, -1072, 2219, 3369,
  5362, 13055, -37814, 3214, 5919, -21900, 2752, -519, 40272, 14539, -11296, 16074, 6962, 27687, -15511, -16910,
  20124, 9804, -7061, -15811, -11578, -19780, 10099, 19504, 18735, -1640, -7741, 18610, 26917, -10452, 22871, -3659,
  -249, -29362, -42021, 8204, 20086, -42493, 5596, -59315, 17176, -18063, 2222, -22745, 8620, 25585, -13025, -19899,
  -13881, -2561, -30018, -3790, 112, 32890, -3830, -2816, -34624, 19370, 6015, 6011, -23303, 30463, -24962, -22262,
  25582, 21552, 18821, 31734, 19334, -20299, -1491, -35170, 29389, 11037, 12101, 597, -35072, 12841, -15524, 6115,
  8715, -7000, -1545, 19131, 2341, -33143, 29218, 3567, 34271, 12984, -13694, 12206, 27479, -29703, 15112, -11610
    };

    /* Copy to Stack */
    memcpy(A, A_init, sizeof(A_init));
    memcpy(B, B_init, sizeof(B_init));
    memcpy(C, C_init, sizeof(C_init));
    memcpy(D_ref, D_ref_init, sizeof(D_ref_init));

    #undef M
    #undef K
    #undef N
      // 初始化 UART
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    // 打印计算结果 C（int32）
    // print_uart("C original:\r\n");
    // print_mat_i32((int32_t*)C,4, 4);
    
    // 计算
    matmul_matrix_i8_i32(A, B, C, DIM_M, DIM_N, DIM_K);
    debug_delay_cycles(1000000);

    print_uart("Matrix A:\r\n");
    print_uart("A address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)A);
    print_uart("\r\n");
    print_mat_i8((const int8_t*)A, DIM_M, DIM_K);

    print_uart("Matrix B (KxN):\r\n");
    print_uart("B address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)B);
    print_uart("\r\n");
    print_mat_i8((const int8_t*)B, DIM_K, DIM_N);

    // 打印计算结果 C（int32）
    print_uart("C result:\r\n");
    print_uart("C address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)C);
    print_uart("\r\n");
    print_mat_i32((int32_t*)C, 16, 16);

    print_uart("Golden D_ref (int32):\r\n");
    print_uart("D_ref address:\r\n ");
    print_uart_hex((uint64_t)(uintptr_t)D_ref);
    print_uart("\r\n");
    print_mat_i32((const int32_t*)D_ref, 16, 16);

    int mismatch = 0;
    for (int idx = 0; idx < 16 * 16; ++idx) {
      if (C[idx] != D_ref[idx]) {
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


    // printf("C result:\n");
    // for (int i = 0; i < DIM_M; i++) {
    //     for (int j = 0; j < DIM_N; j++) {
    //         printf("%d ", C[i * DIM_N + j]);
    //     }
    //     printf("\n");
    
    //   }
    // // 与 golden (D_i32) 对比
    // int mismatch = 0;
    // for (int idx = 0; idx < DIM_M * DIM_N; ++idx) {
    //     if (C[idx] != D_i32[idx]) {
    //         if (mismatch < 16) { // 只打印前16个差异
    //             int r = idx / DIM_N;
    //             int c = idx % DIM_N;
    //             printf("mismatch (%d,%d): got %d expected %d\n",
    //                    r, c, C[idx], D_i32[idx]);
    //         }
    //         mismatch++;
    //     }
    // }
    // if (mismatch == 0) {
    //     printf("Compare with D: PASS\n");
    // } else {
    //     printf("Compare with D: FAIL, total mismatches=%d\n", mismatch);
    // }


    return 0;
}

  #endif
