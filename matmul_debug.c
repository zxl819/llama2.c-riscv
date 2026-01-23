
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

// 标准矩阵乘法实现（参考实现）
static void matmul_reference(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


#include <stdio.h> 
static inline void matmul_vector2(float* xout, float* x, float* w, int n, int d) { 
    print_uart("enter matmul:\r\n"); 
    static float temp[2048] __attribute__((aligned(64)));
    
    // Manually zero temp 
    for (int k = 0; k < 2048; k++) temp[k] = 0.0f;

    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* row = w + i * n;
        //for (int j = 0; j < n; j++)

        float* ptr_x = x;
        float acc = 0.0f;  // 标量累加器
        size_t avl = n;
        while (avl > 0) {
             size_t vl = __riscv_vsetvl_e32m1(avl);

             vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row, vl);

             vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl); 
             vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            __riscv_vse32_v_f32m1(temp, vprod, vl);
            debug_delay_cycles(50);
            //volatile float* vtemp = temp;
            // 标量累加：将临时数组中的元素累加到acc
            for (size_t j = 0; j < vl; j++) {
                acc += temp[j];
            }
            row += vl;
            ptr_x += vl;
            avl -= vl;
        }
        xout[i] = acc;
        // val += row[j] * x[j];
        // xout[i] = val;
    }
}
/**
 * @brief 方法二：使用更宽的向量和归约指令实现矩阵-向量乘法 (fp32)
 *
 * @param xout 输出向量 (d x 1)
 * @param x    输入向量 (n x 1)
 * @param w    输入矩阵 (d x n)
 * @param n    向量x的维度和矩阵w的列数
 * @param d    矩阵w的行数和输出向量xout的维度
 */
static void __attribute__((noinline)) matmul_vector(float* xout, float* x, float* w, int n, int d, float * debug_vec_a, float * debug_vec_b,float * debug_vec_c, float * debug_vec_d) {
    print_uart("enter matmul_vector (fp32) with FULL DEBUG:\r\n");
    static float temp[2048] __attribute__((aligned(64)));
    print_uart("DEBUG: Initial w pointer: 0x");
    print_uart_hex((unsigned long long)w);
    print_uart("\r\n");
    
    print_uart("DEBUG: Initial x pointer: 0x");
    print_uart_hex((unsigned long long)x);
    print_uart("\r\n");
    
    // Clear temp for safety
    for(int k=0; k<2048; k++) temp[k] = 0.0f;

    for (int i = 0; i < d; i++) {
        print_uart("\r\n--- Row "); print_uart_int_dec(i); print_uart(" ---\r\n");
        
        float* row = w + i * n;
        float* ptr_x = x;
        size_t avl = n;
        size_t temp_idx = 0;
        int val = 0;
        size_t block_offset_in_row = 0; 
        
        float ref_row_acc = 0.0f; // Reference accumulator for this row

        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            print_uart("DEBUG: About to vle32.v from row ptr: 0x");
            print_uart_hex((unsigned long long)row);
            print_uart("\r\n");

            // === 调试点 3: 检查即将加载的内存内容 ===
            print_uart("DEBUG: About to load from row. First 4 elements: ");
            for (int k = 0; k < 4; ++k) {
                print_uart_float(row[k]);
                print_uart(" ");
            }
            print_uart("\r\n");

            print_uart("DEBUG: About to load from ptr_x. First 4 elements: ");
            for (int k = 0; k < 4; ++k) {
                print_uart_float(ptr_x[k]);
                print_uart(" ");
            }
            print_uart("\r\n");
            
            print_uart("  Chunk: vl="); print_uart_int_dec(vl); 
            print_uart(", avl="); print_uart_int_dec(avl); print_uart("\r\n");

            // Load vectors
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row, vl);
            print_uart("DEBUG: About to vle32.v from ptr_x ptr: 0x");
            print_uart_hex((unsigned long long)ptr_x);
            print_uart("\r\n");

            // 计算目标地址：基地址 + 行偏移 + 块偏移
            float* dest_ptr = &debug_vec_a[i * n + block_offset_in_row];
            // 将向量数据存储到计算出的地址
            __riscv_vse32_v_f32m1(dest_ptr, vrow, vl);
            debug_delay_cycles(50);
            print_uart("    Loaded w (vec): ");
            for (size_t k = 0; k < vl; k++) {
            print_uart_float(dest_ptr[k]); 
            print_uart(" ");
            }
            print_uart("\r\n");


            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl);
            float* dest_ptr_b = &debug_vec_b[i * n + block_offset_in_row]; 
            __riscv_vse32_v_f32m1(dest_ptr_b, vx, vl);
             debug_delay_cycles(50);
            print_uart("    Loaded x: ");
            for (size_t k = 0; k < vl; k++) {
                print_uart_float(debug_vec_b[k]);
                print_uart(" ");
            }
            print_uart("\r\n");


            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            float* dest_ptr_c = &debug_vec_c[i * n + block_offset_in_row];
            __riscv_vse32_v_f32m1(dest_ptr_c, vprod, vl);
            debug_delay_cycles(50);
            print_uart("    Vector Prod: ");
            // 计算目标地址：基地址 + 行偏移 + 块偏移
            
            for (size_t k = 0; k < vl; k++) {
            print_uart_float(dest_ptr_c[k]); // <--- 修改这里
            print_uart(" ");
            }
            print_uart("\r\n");

            // Store vprod to temp
            float* dest_ptr_d = &debug_vec_d[i * n + block_offset_in_row];
            __riscv_vse32_v_f32m1(dest_ptr_d, vprod, vl);
            debug_delay_cycles(50);
            for (size_t k = 0; k < vl; k++) {
                float r_val = row[k];
                float x_val = ptr_x[k];
                float vec_prod = dest_ptr_d[k];
                float ref_prod = r_val * x_val;
                ref_row_acc += ref_prod;

                print_uart("    Idx "); print_uart_int_dec(temp_idx + k);
                print_uart(": w="); print_uart_float(r_val);
                print_uart(" * x="); print_uart_float(x_val);
                print_uart(" | VecProd="); print_uart_float(vec_prod);
                print_uart(" | RefProd="); print_uart_float(ref_prod);
                
                if (!float_equals(vec_prod, ref_prod, 0.0001f)) {
                     print_uart(" [MISMATCH]\r\n");
                } else {
                     print_uart(" [OK]\r\n");
                }
            }
            print_uart("\r\n");
            
            
            row += vl;
            ptr_x += vl;
            temp_idx += vl;  // 更新 temp 数组中的位置
            block_offset_in_row += vl;
            avl -= vl;
        }
        
        // Accumulate vector results
        float acc = 0.0f;
        // 计算当前行在 debug_vec_d 中的起始地址
        float* current_row_products = &debug_vec_d[i * n];
        // print_uart("  Accumulating temp array:\r\n");
        for (size_t j = 0; j < n; j++) {
            acc += current_row_products[j];
            print_uart("acc");
            print_uart_int_dec(j);
            print_uart("=");
            print_uart_float(acc);
            print_uart("\r\n");
        }
        
        xout[i] = acc;
        
        print_uart("  Row Result: VecAcc="); print_uart_float(acc);
        print_uart(" | RefAcc="); print_uart_float(ref_row_acc);
        if (!float_equals(acc, ref_row_acc, 0.001f)) {
             print_uart(" [ROW MISMATCH]\r\n");
        } else {
             print_uart(" [ROW OK]\r\n");
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

    // 1. 正确分配 w 为一个 d x n 的二维矩阵
    float w[d][n];
    float debug_vec_a[d][n];
    float debug_vec_b[d][n];
    float debug_vec_c[d][n];
    float debug_vec_d[d][n];
    // 2. 正确分配 x 为一个长度为 n 的向量
    float x[n];
    // 3. 为结果分配 xout，长度为 d
    float xout[d];
     // 4. 为参考结果分配 expected，长度为 d
    float expected[d];
    print_uart("initializing data:\r\n");
    // 初始化 w 和 x
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            w[i][j] = (float)(i + j); // 给一个示例值
        }
    }
    for (int k = 0; k < n; k++) {
        x[k] = (float)k;
    }

    print_uart("Input Vector x:\r\n");
    for (int k = 0; k < n; k++) {
        print_uart_float(x[k]);
        print_uart(" ");
        if ((k + 1) % 8 == 0) print_uart("\r\n");
    }
    print_uart("\r\n");

    print_uart("Input Matrix w:\r\n");
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            print_uart_float(w[i][j]);
            print_uart(" ");
        }
        print_uart("\r\n");
    }

    // 4. 正确调用函数
    // 注意：传递二维数组 w 时，我们传递它的首地址 &w[0][0]
    print_uart("matmul initializing:\r\n");
    matmul_vector(xout, x, &w[0][0], n, d,debug_vec_a,debug_vec_b,debug_vec_c,debug_vec_d);

    print_uart("Full xout result:\r\n");
    for (int i = 0; i < d; i++) {
    print_uart_float(xout[i]); // 打印数值
    print_uart(" ");           // 打印一个空格分隔

    // 每 8 个元素换一次行
    if ((i + 1) % 8 == 0) {
        print_uart("\r\n");
    }
}
 //计算预期结果
    print_uart("\r\nCalculating expected results:\r\n");
    matmul_reference(expected, x, &w[0][0], n, d);
    
    // 打印预期结果
    print_uart("Expected results:\r\n");
    for (int i = 0; i < d; i++) {
        print_uart_float(expected[i]);
        print_uart(" ");
        
        // 每 8 个元素换一次行
        if ((i + 1) % 8 == 0) {
            print_uart("\r\n");
        }
    }
    
    // 比较结果
    print_uart("\r\nComparing results:\r\n");
    float epsilon = 0.0001f; // 允许的误差范围
    int is_correct = compare_arrays(xout, expected, d, epsilon);
    
    if (is_correct) {
        print_uart("TEST PASSED: Results match expected values within tolerance.\r\n");
    } else {
        print_uart("TEST FAILED: Results do not match expected values.\r\n");
        
        // 打印不匹配的元素
        print_uart("Mismatched elements:\r\n");
        for (int i = 0; i < d; i++) {
            if (!float_equals(xout[i], expected[i], epsilon)) {
                print_uart("Index ");
                print_uart_int_dec(i);
                print_uart(": got ");
                print_uart_float(xout[i]);
                print_uart(", expected ");
                print_uart_float(expected[i]);
                print_uart("\r\n");
            }
        }
    }

//     // 确保在最后换行，以防最后一行不足7个
    print_uart("\r\n");

}