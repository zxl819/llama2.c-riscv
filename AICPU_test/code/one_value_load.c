#include <stdint.h>
#include <stddef.h>

#ifndef MATMUL_MATRIX_KERNEL_ONLY
#include "uart_helper.c"
#endif

#define CLOCK_FREQUENCY 24000000
#define UART_BITRATE    115200

// AME inline-asm: msettilem a0, a0
static inline int asm_msettilem(int rem) {
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04055577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}

// AME inline-asm: msettilen a0, a0
static inline int asm_msettilen(int rem) {
  register size_t a0 asm("a0") = (size_t)rem;
  asm volatile(".word 0x04054577\n\t# %0" : "+r"(a0) :: "memory");
  return (int)a0;
}

// AME inline-asm: mlce32.m acc0, (a0), a1
static inline void asm_mlce32_acc0(const int32_t *base, int stride_bytes) {
  register const int32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x00b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

// AME inline-asm: msce32.m acc0, (a0), a1
static inline void asm_msce32_acc0(int32_t *base, int stride_bytes) {
  register int32_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x02b52077\n\t# %0 %1" :: "r"(a0), "r"(a1) : "memory");
}

// 简单的空转延时函数，使用 asm volatile 防止被优化掉
static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}

// 最小 AME load/store 示例：把一个数组元素经由 acc0 搬运到另一个数组。
__attribute__((aligned(64))) static const int32_t g_src[] = {
  11, 22, 33, 44, 55
};

__attribute__((aligned(64))) static int32_t g_dst[] = {
  0, 0, 0, 0, 0
};

int main(void) {
  const int idx = 2;
  const int stride_bytes = (int)sizeof(int32_t);

  init_uart(CLOCK_FREQUENCY, UART_BITRATE);

  // 1x1  AME load/store。
  (void)asm_msettilem(1);
  (void)asm_msettilen(1);

  print_uart("AME load/store one value test\r\n");
  print_uart("before dst=");
  print_dec32(g_dst[idx]);
  print_uart("\r\n");

  print_uart("load addr(src)=0x");
  print_uart_hex((uint64_t)(uintptr_t)(&g_src[idx]));
  print_uart("\r\n");
  print_uart("store addr(dst)=0x");
  print_uart_hex((uint64_t)(uintptr_t)(&g_dst[idx]));
  print_uart("\r\n");

  // load: memory -> acc0
  asm_mlce32_acc0(&g_src[idx], stride_bytes);
  // store: acc0 -> memory
  asm_msce32_acc0(&g_dst[idx], stride_bytes);

  debug_delay_cycles(1000); // 确保 store 完成
  print_uart("idx=");
  print_dec32(idx);
  print_uart(" value=");
  print_dec32(g_dst[idx]);
  print_uart("\r\n");

  return 0;
}
