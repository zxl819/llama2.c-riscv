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
//#include "data.h"
#include "uart_helper.c"
// Standalone build historically included uart.c directly.
// Keep it ONLY for the standalone demo to avoid multiple-definition issues when
// linking into other programs that also compile uart.c.
//#include "uart.c"
#endif

//A*B

#ifndef MATMUL_MATRIX_KERNEL_ONLY
//用于UART输出调试信息
#define CLOCK_FREQUENCY 50000000 //50MHz
#define UART_BITRATE    115200
#endif

// #define DIM_M 2
// #define DIM_K 2
// #define DIM_N 2
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
  // #ifndef MATMUL_MATRIX_LDC_ELEMS
  // #define MATMUL_MATRIX_LDC_ELEMS 16
  // #endif
  #ifndef MATMUL_MATRIX_LDC_ELEMS
  #define MATMUL_MATRIX_LDC_ELEMS 4
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
    /* Include data.h locally to ensure A, B, C, D_ref are allocated on the stack */
    #include "data_2x2.h"

      // 初始化 UART
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    // 打印计算结果 C（int32）
    // print_uart("C original:\r\n");
    // print_mat_i32((int32_t*)C,4, 4);
    
    // 计算
    matmul_matrix_i8_i32(A, B, C, DIM_M, DIM_N, DIM_K);
    debug_delay_cycles(1000000);

    print_uart("Matrix A:\r\n");
    // print_uart("A address:\r\n ");
    // print_uart_hex((uint64_t)(uintptr_t)A);
    // print_uart("\r\n");
    print_mat_i8((const int8_t*)A, DIM_M, DIM_K);

    print_uart("Matrix B (KxN):\r\n");
    // print_uart("B address:\r\n ");
    // print_uart_hex((uint64_t)(uintptr_t)B);
    // print_uart("\r\n");
    print_mat_i8((const int8_t*)B, DIM_K, DIM_N);

    // 打印计算结果 C（int32）
    print_uart("C result:\r\n");
    // print_uart("C address:\r\n ");
    // print_uart_hex((uint64_t)(uintptr_t)C);
    // print_uart("\r\n");
    print_mat_i32((int32_t*)C, 4, 4);

    print_uart("Golden D_ref (int32):\r\n");
    // print_uart("D_ref address:\r\n ");
    // print_uart_hex((uint64_t)(uintptr_t)D_ref);
    // print_uart("\r\n");
    print_mat_i32((const int32_t*)D_ref, 4, 4);

    int mismatch = 0;
    for (int idx = 0; idx < 4 * 4; ++idx) {
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
