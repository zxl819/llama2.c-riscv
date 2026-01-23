
#include <riscv_vector.h>
#include <stdio.h>
#include "uart.h"
#include "uart.c"

#define CLOCK_FREQUENCY 50000000 //50MHz
#define UART_BITRATE    115200  
// 简单的空转延时函数，使用 asm volatile 防止被优化掉
static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}

// // 简单 UART 输出工具
static inline void uart_putc(char c) { write_serial((uint8_t)c); }

static void print_dec32(int32_t v) {
    char buf[16]; int i = 0;
    if (v == 0) { uart_putc('0'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    while (v && i < (int)sizeof(buf)) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i--) uart_putc(buf[i]);
}

static void print_i32_array(const char* name, const int32_t* a, int n) {
    print_uart(name);
    print_uart(":\r\n");
    for (int i = 0; i < n; ++i) {
        print_dec32(a[i]);
        if (i + 1 < n) uart_putc(' ');
    }
    print_uart("\r\n");
}

// Helper for printing numbers (decimal)
static void print_uart_int_dec(uint64_t val) {
    char buf[32];
    int i = 0;
    if (val == 0) {
        print_uart("0");
        return;
    }
    while (val > 0) {
        buf[i++] = (val % 10) + '0';
        val /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        write_serial(buf[j]);
    }
}

// Helper for printing floats (simple)
static void print_uart_float(float val) {
    if (val < 0) {
        print_uart("-");
        val = -val;
    }
    uint64_t int_part = (uint64_t)val;
    float frac_part = val - int_part;
    print_uart_int_dec(int_part);
    print_uart(".");
    uint64_t frac_int = (uint64_t)(frac_part * 1000000);
    print_uart_int_dec(frac_int);
}
static void print_f32_array(const char* name, const float* a, int n) {
    print_uart(name);
    print_uart(":\r\n");
    for (int i = 0; i < n; ++i) {
        print_uart_float(a[i]);
        if (i + 1 < n) uart_putc(' ');
    }
    print_uart("\r\n");
}

int main() {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    
    // 定义向量长度
    size_t vl = 16;  // 向量长度，根据硬件和RVV版本调整
    const int n = 16;

    // 创建两个向量
    vfloat32m1_t a, b, result;
    
    // 初始化向量数据
    float data_a[n] = {1, 2, 3, 4, 5, 6, 7, 8 ,3,3,3,3,3,3,3,3};
    float data_b[n] = {2, 4, 6, 8, 10, 12, 14, 16,3,3,3,3,3,3,3,3};
    float debug_a[n]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    float debug_b[n]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    // 输出结果
    float result_data[n]={2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    
    // 将数据加载到向量中
    a = __riscv_vle32_v_f32m1(data_a, vl);
    debug_delay_cycles(1000000); //debug
    __riscv_vse32_v_f32m1(debug_a, a, vl);
    debug_delay_cycles(1000000);
    print_uart("\r\n a load:\r\n");
    print_f32_array("a load", debug_a, n);
    debug_delay_cycles(1000000); //debug



    b = __riscv_vle32_v_f32m1(data_b, vl);
    debug_delay_cycles(1000000); //debug
    __riscv_vse32_v_f32m1(debug_a, b, vl);
    debug_delay_cycles(1000000);
    print_uart("\r\n b load:\r\n");
    debug_delay_cycles(1000000); //debug
    print_f32_array("b load", debug_a, n);
    debug_delay_cycles(1000000); //debug
    

    // 执行加法
    result = __riscv_vfadd_vv_f32m1(a, b, vl);
    debug_delay_cycles(1000000);//debug
    

    // ==============UART 打印结果====================
    __riscv_vse32_v_f32m1(result_data, result, vl);

    debug_delay_cycles(1000000);
    print_uart("Result UART:\r\n");
    debug_delay_cycles(1000000); //debug
    print_f32_array("data_a", data_a, n);
    debug_delay_cycles(1000000); //debug
    print_f32_array("data_b", data_b, n);
    debug_delay_cycles(1000000); //debug
    print_f32_array("result", result_data, n);

    
    
    // //=======SPIKE===============
    // printf("Result: ");
    // for (int i = 0; i < 8; i++) {
    //     printf("%d ", result_data[i]);
    // }
    // printf("\n");

    return 0;
}
