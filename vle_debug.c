
#include <riscv_vector.h>
#include "uart.h"
#include "uart.c"

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
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

static inline float my_fabs(float x) {
    return x < 0.0f ? -x : x;
}
void print_uart_hex(unsigned long long val) {
    char hex_chars[] = "0123456789ABCDEF";
    char buf[17];
    buf[16] = '\0';
    for(int i = 15; i >= 0; --i) {
        buf[i] = hex_chars[val & 0xF];
        val >>= 4;
    }
    print_uart(buf);
}
// 比较两个浮点数是否近似相等
static int float_equals(float a, float b, float epsilon) {
    // 使用我们自己的 my_fabs
    return my_fabs(a - b) < epsilon;
}
// 比较两个浮点数组是否近似相等
static int compare_arrays(float* a, float* b, int size, float epsilon) {
    for (int i = 0; i < size; i++) {
        if (!float_equals(a[i], b[i], epsilon)) {
            return 0; // 不相等
        }
    }
    return 1; // 相等
}
static void __attribute__((noinline)) vle32debug(float* xout, float* w1, float* w2, int n, int d) {
    print_uart("enter vle32 with FULL DEBUG:\r\n");
    static float temp[2048] __attribute__((aligned(64)));
    print_uart("DEBUG: Initial w pointer: 0x");
    print_uart_hex((unsigned long long)w1);
    print_uart("\r\n");
    
    print_uart("DEBUG: Initial x pointer: 0x");
    print_uart_hex((unsigned long long)w2);
    print_uart("\r\n");
    
    // Clear temp for safety
    for(int k=0; k<2048; k++) temp[k] = 0.0f;

    for (int i = 0; i < d; i++) {
        print_uart("\r\n--- Row "); print_uart_int_dec(i); print_uart(" ---\r\n");
        
        float* row1 = w1 + i * n;
        float* row2 = w2 + i * n;
        size_t avl = n;
        size_t temp_idx = 0;
        int val = 0;
        
        float ref_row_acc = 0.0f; // Reference accumulator for this row

        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            print_uart("DEBUG: About to vle32.v from row1 ptr: 0x");
            print_uart_hex((unsigned long long)row1);
            print_uart("\r\n");

            // === 调试点 3: 检查即将加载的内存内容 ===
            print_uart("DEBUG: About to load from row1. First 4 elements: ");
            for (int k = 0; k < 4; ++k) {
                print_uart_float(row1[k]);
                print_uart(" ");
            }
            print_uart("\r\n");

            print_uart("DEBUG: About to load from row2. First 4 elements: ");
            for (int k = 0; k < 4; ++k) {
                print_uart_float(row2[k]);
                print_uart(" ");
            }
            print_uart("\r\n");
            
            print_uart("  Chunk: vl="); print_uart_int_dec(vl); 
            print_uart(", avl="); print_uart_int_dec(avl); print_uart("\r\n");

            // Load vectors
            // 强制使用 v8
            asm volatile("vle32.v v8, (%0)" : : "r"(row1));
            
            // 2. Load vrow2 (Force compiler to use a different register by keeping vrow1 live)
            print_uart("DEBUG: Loading vrow2 from: 0x"); print_uart_hex((unsigned long long)row2); print_uart("\r\n");
            // 强制使用 v9
            asm volatile("vle32.v v9, (%0)" : : "r"(row2));

            debug_delay_cycles(50);

            // 3. Verify vrow1 (vrow1 is used here, so it must be alive during vrow2 load)
            static float debug_vec[16]; 
            asm volatile("vse32.v v8, (%0)" : : "r"(debug_vec));
            
            print_uart("    Checking vrow1 (vec vs mem): \r\n");
            for (size_t k = 0; k < vl; k++) {
                float vec_val = debug_vec[k];
                float mem_val = row1[k];
                if (!float_equals(vec_val, mem_val, 0.0001f)) {
                     print_uart("      [MISMATCH] vrow1["); print_uart_int_dec(k); print_uart("]\r\n");
                }
            }

            // 4. Verify vrow2
            asm volatile("vse32.v v9, (%0)" : : "r"(debug_vec));
            
            print_uart("    Checking vrow2 (vec vs mem): \r\n");
            for (size_t k = 0; k < vl; k++) {
                float vec_val = debug_vec[k];
                float mem_val = row2[k];
                print_uart("      ["); print_uart_int_dec(k); print_uart("] Vec: ");
                print_uart_float(vec_val);
                print_uart(" Mem: ");
                print_uart_float(mem_val);
                
                if (!float_equals(vec_val, mem_val, 0.0001f)) {
                     print_uart(" [MISMATCH]");
                } else {
                     print_uart(" [OK]");
                }
                print_uart("\r\n");
            }

            
            // Store vrow1 to temp and verify
            // __riscv_vse32_v_f32m1(&temp[temp_idx], vrow1, vl);
            asm volatile("vse32.v v8, (%0)" : : "r"(&temp[temp_idx]));
            debug_delay_cycles(50);
            
            // Verify this chunk
            volatile float* vtemp = &temp[temp_idx];
            
            for (size_t k = 0; k < vl; k++) {
                float stored_val = temp[temp_idx + k];
                float ref_val = row1[k];
                print_uart("      ["); print_uart_int_dec(k); print_uart("] Stored: ");
                print_uart_float(stored_val);
                print_uart(" Ref: ");
                print_uart_float(ref_val);
                if (!float_equals(stored_val, ref_val, 0.0001f)) {
                     print_uart(" [MISMATCH]\r\n");
                } else {
                     print_uart(" [OK]\r\n");
                }
            }

            // Store vrow2 to temp and verify
            // __riscv_vse32_v_f32m1(&temp[temp_idx], vrow2, vl);
            asm volatile("vse32.v v9, (%0)" : : "r"(&temp[temp_idx]));
            debug_delay_cycles(50);

            print_uart("    Verifying vrow2 store:\r\n");
            for (size_t k = 0; k < vl; k++) {
                float stored_val = temp[temp_idx + k];
                float ref_val = row2[k];
                print_uart("      ["); print_uart_int_dec(k); print_uart("] Stored: ");
                print_uart_float(stored_val);
                print_uart(" Ref: ");
                print_uart_float(ref_val);
                if (!float_equals(stored_val, ref_val, 0.0001f)) {
                     print_uart(" [MISMATCH]\r\n");
                } else {
                     print_uart(" [OK]\r\n");
                }
            }

            row1 += vl;
            row2 += vl;
            temp_idx += vl;
            avl -= vl;
        }
}
}
int real_main() __attribute__((no_vector));

__attribute__((naked)) void main() {
    asm volatile (
        "li t0, 0x6600\n"       // 准备掩码 (VS=11, FS=11)
        "csrs mstatus, t0\n"   // 开启 VS 和 FS
        "j real_main\n"        // 跳转到 C 函数
    );
}

int real_main() {
    // Enable VS extension (bits 9-10 of mstatus)
    // unsigned long mstatus;
    // asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    // mstatus |= 0x200; 
    // asm volatile("csrw mstatus, %0" :: "r"(mstatus));
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("enter main:\r\n");
    int n = 16;
    int d = 16;

    // 1. 正确分配 w1 为一个 d x n 的二维矩阵
    float w1[d][n];
    // 2. 正确分配 w2 为一个 d x n 的二维矩阵
    float w2[d][n];
    // 3. 为结果分配 xout
    float xout[d];

    print_uart("initializing data:\r\n");
    // 初始化 w1 和 w2
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            w1[i][j] = (float)(i + j); // 给一个示例值
        }
    }
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            w2[i][j] = (float)(i * j); // 给一个示例值
        }
    }

    print_uart("Input Matrix w1:\r\n");
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            print_uart_float(w1[i][j]);
            print_uart(" ");
        }
        print_uart("\r\n");
    }

    print_uart("Input Matrix w2:\r\n");
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            print_uart_float(w2[i][j]);
            print_uart(" ");
        }
        print_uart("\r\n");
    }

    // 4. 正确调用函数
    // 注意：传递二维数组 w 时，我们传递它的首地址 &w1[0][0] &w2[0][0]
    print_uart("vledebug initialization:\r\n");
    vle32debug(xout, &w1[0][0], &w2[0][0], n, d);

}
