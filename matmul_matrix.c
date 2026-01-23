#include <stdint.h>
#include <riscv_matrix.h>
#include <stddef.h>
#include "matrix/matrix_intrinsic.h"

// If you want to link this file into a larger program (e.g. RVV llama2 model),
// define MATMUL_MATRIX_KERNEL_ONLY=1 for this translation unit.
// That disables the standalone demo (main/UART/data globals) and only exposes
// a callable Matrix-kernel function.
#ifndef MATMUL_MATRIX_KERNEL_ONLY
#include <string.h>
#include "data.h"
#include "uart.h"
// Standalone build historically included uart.c directly.
// Keep it ONLY for the standalone demo to avoid multiple-definition issues when
// linking into other programs that also compile uart.c.
#include "uart.c"
#endif

//A*B

#ifndef MATMUL_MATRIX_KERNEL_ONLY
//用于UART输出调试信息
#define CLOCK_FREQUENCY 50000000 //50MHz
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
__attribute__((section(".matA"), aligned(64))) int8_t  A[DIM_M*DIM_K];
__attribute__((section(".matB"), aligned(64))) int8_t  B[DIM_K*DIM_N];
__attribute__((section(".matC"), aligned(64))) int32_t C[DIM_M*DIM_N];
#endif

static inline int matmul_batch1(const int8_t *A, const int8_t *B, int32_t *C,
                                 int m, int n, int k) {
  // lda, ldb, ldc in bytes for msetilem/len/k                                
  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = n * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(int32_t);


  const int dataSize = sizeof(int8_t);
  // matmul
  int tile_m = 0, tile_n = 0, tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    tile_m = msettilem(m - i);
    //printf("i = %d\n", i);  // 添加调试
    //printf("debug: i=%d remaining=%d tile_m=%d\n", i, m - i, tile_m);
    for (int j = 0; j < n; j += tile_n) {
      tile_n = msettilen(n - j);
      //printf("debug: j=%d remaining=%d tile_n=%d\n", j, n - j, tile_n);
      // SET_MBA0_I32();
      //mint32_t acc1;
      //initialize acc using C

      mint32_t acc1 = mlc_m(C + i * n + j, ldc_bytes);
      
      for (int kk = 0; kk < k; kk += tile_k) {
        tile_k = msettilek(k - kk);
        mint8_t tr0 = mla_m(A + i * k + kk, lda_bytes);
        //mint8_t tr1 = mlbt_m(B + kk * k + j, ldb_bytes); //AB+C
        mint8_t tr1 = mlb_m(B + j * n + kk, ldb_bytes); //ABT+C
        //acc = mqma_mm(acc, tr0, tr1);
        acc1 = mqma_b_mm(acc1, tr0, tr1);
        //printf("debug: kk=%d remaining=%d tile_k=%d\n", kk, k - kk, tile_k);
      }
      msc_m(acc1, C + i * n + j, ldc_bytes);
    }
    //printf("debug: i=%d remaining=%d tile_m=%d\n", i, m - i, tile_m);
  }

  // debug_delay_cycles(1000);
    for (int i = 0; i < m; i += tile_m) {
    tile_m = msettilem(m - i);
    //printf("i = %d\n", i);  // 添加调试
    for (int j = 0; j < n; j += tile_n) {
      tile_n = msettilen(n - j);
      // SET_MBA0_I32();
      //mint32_t acc1;
      //initialize acc using C
      mint32_t acc1 = mlc_m(C + i * n + j, ldc_bytes);
      msc_m(acc1, C + i * n + j, ldc_bytes);
     }
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
// 简单输出 1 个字符到 UART
static inline void uart_putc(char c) {
  write_serial((uint8_t)c);
}

// 打印有符号 32 位十进制
static void print_dec32(int32_t v) {
    char buf[16];
    int i = 0;
    if (v == 0) { uart_putc('0'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    while (v && i < (int)sizeof(buf)) {
        buf[i++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (i--) uart_putc(buf[i]);
}

// 打印 int32 矩阵，rows x cols，每行结尾换行
static void print_mat_i32(const int32_t* m, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            print_dec32(m[r * cols + c]);
            if (c + 1 < cols) uart_putc(' ');
        }
        print_uart("\r\n");
    }
}

static inline void print_dec8(int8_t v) {
    print_dec32((int32_t)v);
}

static void print_mat_i8(const int8_t* m, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            print_dec8(m[r * cols + c]);
            if (c + 1 < cols) uart_putc(' ');
        }
        print_uart("\r\n");
    }
}

int main()
{
    // 将 data.h 中的 A_i8/B_i8 拷贝到固定段 .matA/.matB；C 清零
    memcpy(A, A_i8, M * K * sizeof(int8_t));
    memcpy(B, B_i8, K * N * sizeof(int8_t));
    memcpy(C, C_i32, M * N * sizeof(int32_t));

    // 计算
    matmul_matrix_i8_i32(A, B, C, DIM_M, DIM_N, DIM_K);
    debug_delay_cycles(1000000);
    // 初始化 UART
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);

    print_uart("Matrix A:\r\n");
    print_mat_i8((const int8_t*)A, DIM_M, DIM_K);

    print_uart("Matrix B:\r\n");
    print_mat_i8((const int8_t*)B, DIM_K, DIM_N);

    // 打印计算结果 C（int32）
    print_uart("C result:\r\n");
    print_mat_i32(C, DIM_M, DIM_N);

    // 打印 golden D（int32）
    print_uart("Golden D_i32 (int32):\r\n");
    print_mat_i32((const int32_t*)D_i32, DIM_M, DIM_N);


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
