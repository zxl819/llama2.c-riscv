#include <stdint.h>
#include <stddef.h>

// For human-readable prints (does not affect the Matrix kernel path).
#include "fp8_e4m3fn.h"

#ifndef MATMUL_MATRIX_KERNEL_ONLY
#include <string.h>
#include "uart_helper.c"
#endif

#ifndef MATMUL_MATRIX_KERNEL_ONLY
#define CLOCK_FREQUENCY 24000000 // 50MHz
#define UART_BITRATE    115200
#endif

// DIM_M, DIM_K, DIM_N are now provided by data_16x16_fp8.h
// but kept as fallbacks if needed.
#ifndef DIM_M
#define DIM_M 16
#define DIM_K 16
#define DIM_N 16
#endif

static inline void debug_delay_cycles(unsigned cycles) {
#if defined(__riscv)
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
#else
  (void)cycles;
#endif
}

static inline float f32_abs(float x) {
  return (x < 0.0f) ? -x : x;
}

#ifndef F32_PRINT_FRAC_DIGITS
#define F32_PRINT_FRAC_DIGITS 4
#endif

static void print_u32_zpad(uint32_t v, int width) {
  char buf[10];
  if (width > (int)sizeof(buf)) width = (int)sizeof(buf);
  for (int i = width - 1; i >= 0; --i) {
    buf[i] = (char)('0' + (v % 10u));
    v /= 10u;
  }
  for (int i = 0; i < width; ++i) {
    write_serial((uint8_t)buf[i]);
  }
}

static void print_f32_fixed(float x) {
  if (x != x) { print_uart("nan"); return; }
  if (x < 0.0f) { write_serial((uint8_t)'-'); x = -x; }
  int32_t ip = (int32_t)x;
  float frac = x - (float)ip;
  uint32_t pow10 = 1;
  for (int i = 0; i < F32_PRINT_FRAC_DIGITS; ++i) pow10 *= 10u;
  uint32_t fp = (uint32_t)(frac * (float)pow10 + 0.5f);
  if (fp >= pow10) { ip += 1; fp = 0; }
  print_dec32(ip);
  write_serial((uint8_t)'.');
  print_u32_zpad(fp, F32_PRINT_FRAC_DIGITS);
}

static void print_mat_fp8_e4m3fn_as_f32(const uint8_t* m, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float f = fp8e4m3fn_to_f32(m[r * cols + c]);
      print_f32_fixed(f);
      if (c + 1 < cols) write_serial((uint8_t)' ');
    }
    print_uart("\r\n");
  }
}

static void print_mat_f32(const float* m, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      print_f32_fixed(m[r * cols + c]);
      if (c + 1 < cols) write_serial((uint8_t)' ');
    }
    print_uart("\r\n");
  }
}

// ---------------------------------------------------------------------------
// Matrix inline-asm helpers
static inline int asm_msettilem(int rem) {
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04055577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}
static inline int asm_msettilen(int rem) {
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04054577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}
static inline int asm_msettilek(int rem) {
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04056577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}
static inline void asm_mlce32_acc0(const uint32_t *base, int stride_bytes) {
  register const uint32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x00b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}
static inline void asm_msce32_acc0(uint32_t *base, int stride_bytes) {
  register uint32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x02b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}
static inline void asm_mlae8_tr0(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}
static inline void asm_mlbe8_tr1(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b500f7\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}
static inline void asm_mfqma_cf_acc0_tr0_tr1(void) {
  asm volatile(".word 0x2A180877" ::: "memory");
}

static inline int matmul_batch1(const uint8_t *A, const uint8_t *B, float *C,
                                int m, int n, int k) {
  const int lda_bytes = k * (int)sizeof(uint8_t);
  const int ldb_bytes = n * (int)sizeof(uint8_t);
#ifndef MATMUL_MATRIX_LDC_ELEMS
#define MATMUL_MATRIX_LDC_ELEMS 16
#endif
  const int ldc_bytes = MATMUL_MATRIX_LDC_ELEMS * (int)sizeof(float);

  int tile_m = 0, tile_n = 0, tile_k = 0;
  for (int i = 0; i < m; i += tile_m) {
    tile_m = asm_msettilem(m - i);
    for (int j = 0; j < n; j += tile_n) {
      tile_n = asm_msettilen(n - j);
      float *cptr = C + i * MATMUL_MATRIX_LDC_ELEMS + j;
      asm_mlce32_acc0((const uint32_t *)cptr, ldc_bytes);
      for (int kk = 0; kk < k; kk += tile_k) {
        tile_k = asm_msettilek(k - kk);
        const uint8_t *aptr = A + i * k + kk;
        const uint8_t *bptr = B + j * k + kk;
        asm_mlae8_tr0(aptr, lda_bytes);
        asm_mlbe8_tr1(bptr, ldb_bytes);
        asm_mfqma_cf_acc0_tr0_tr1();
      }
      asm_msce32_acc0(cptr, ldc_bytes);
    }
  }
  return 0;
}

int matmul_matrix_fp8_fp32(const uint8_t *A, const uint8_t *B, float *C,
                           int m, int n, int k) {
  return matmul_batch1(A, B, C, m, n, k);
}

#ifndef MATMUL_MATRIX_KERNEL_ONLY
int main() {
  /* Include data_16x16_fp8.h locally to ensure constants are available */
  #include "data_16x16_fp8.h"

  init_uart(CLOCK_FREQUENCY, UART_BITRATE);

  // Compute: C = A * B^T + C_init
  matmul_matrix_fp8_fp32(A_fp8, B_fp8, C, DIM_M, DIM_N, DIM_K);
  debug_delay_cycles(1000);
  print_uart("Matrix A:\r\n");
  print_uart("A_fp8 bytes:\r\n");
    for (int i = 0; i < DIM_M * DIM_K; ++i) {
      print_uart("0x");
      print_uart_hex((uint64_t)A_fp8[i]);
      if ((i % DIM_K) == (DIM_K - 1)) print_uart("\r\n");
      else write_serial((uint8_t)' ');
    }

  print_uart("A decoded fp32 (E4M3FN):\r\n");
  print_mat_fp8_e4m3fn_as_f32(A_fp8, DIM_M, DIM_K);

  print_uart("Matrix B (KxN):\r\n");
  print_uart("B_fp8 bytes:\r\n");
    for (int i = 0; i < DIM_K * DIM_N; ++i) {
      print_uart("0x");
      print_uart_hex((uint64_t)B_fp8[i]);
      if ((i % DIM_N) == (DIM_N - 1)) print_uart("\r\n");
      else write_serial((uint8_t)' ');
    }


  print_uart("B decoded fp32 (E4M3FN, interpreted as B^T):\r\n");
  print_mat_fp8_e4m3fn_as_f32(B_fp8, DIM_N, DIM_K);

  print_uart("C result (fp32):\r\n");
  print_mat_f32(C, DIM_M, DIM_N);

  print_uart("Golden D_ref (fp32):\r\n");
  print_mat_f32(D_ref, DIM_M, DIM_N);

  int mismatch = 0;
  const float eps = 1e-3f;
  for (int idx = 0; idx < DIM_M * DIM_N; ++idx) {
    float diff = C[idx] - D_ref[idx];
    if (f32_abs(diff) > eps) mismatch++;
  }

  if (mismatch == 0) {
    print_uart("Compare: PASS\r\n");
  } else {
    print_uart("Compare: FAIL mismatches=");
    print_dec32(mismatch);
    print_uart("\r\n");
  }

  return 0;
}
#endif

