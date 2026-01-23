#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

// Use putchar from bare_syscalls.c
extern int putchar(int ch);
extern void _exit(int code);

static void write_serial(char a) {
    putchar(a);
}

static void print_string(const char *str) {
    const char *cur = &str[0];
    while (*cur != '\0') {
        write_serial(*cur);
        ++cur;
    }
}

static void print_hex(unsigned long long val) {
    char buf[16];
    int i = 0;
    if (val == 0) {
        print_string("0");
        return;
    }
    while (val > 0 && i < 16) {
        int digit = val % 16;
        if (digit < 10) buf[i++] = digit + '0';
        else buf[i++] = digit - 10 + 'a';
        val /= 16;
    }
    print_string("0x");
    for (int j = i - 1; j >= 0; j--) {
        write_serial(buf[j]);
    }
}

static void print_int_dec(long long val) {
    char buf[32];
    int i = 0;
    int sign = 0;
    if (val == 0) {
        print_string("0");
        return;
    }
    if (val < 0) {
        sign = 1;
        val = -val;
    }
    while (val > 0) {
        buf[i++] = (val % 10) + '0';
        val /= 10;
    }
    if (sign) write_serial('-');
    for (int j = i - 1; j >= 0; j--) {
        write_serial(buf[j]);
    }
}

static void print_float(float val) {
    if (val < 0) {
        print_string("-");
        val = -val;
    }
    unsigned long long int_part = (unsigned long long)val;
    float frac_part = val - int_part;
    print_int_dec(int_part);
    print_string(".");
    unsigned long long frac_int = (unsigned long long)(frac_part * 1000000);
    // Pad with zeros if needed
    if (frac_int < 100000) print_string("0");
    if (frac_int < 10000) print_string("0");
    if (frac_int < 1000) print_string("0");
    if (frac_int < 100) print_string("0");
    if (frac_int < 10) print_string("0");
    print_int_dec(frac_int);
}

// --- My printf implementation (supports %f, %x) ---
static void my_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd' || *fmt == 'i') {
                int val = va_arg(args, int);
                print_int_dec(val);
            } else if (*fmt == 'u') {
                unsigned int val = va_arg(args, unsigned int);
                print_int_dec(val);
            } else if (*fmt == 'x' || *fmt == 'p') {
                unsigned long long val = va_arg(args, unsigned long long);
                print_hex(val);
            } else if (*fmt == 'z' && *(fmt+1) == 'u') {
                fmt++;
                size_t val = va_arg(args, size_t);
                print_int_dec((long long)val);
            } else if (*fmt == 'f') {
                double val = va_arg(args, double); // float promotes to double in varargs
                print_float((float)val);
            } else if (*fmt == 's') {
                char *str = va_arg(args, char*);
                print_string(str);
            } else if (*fmt == '%') {
                write_serial('%');
            } else {
                write_serial('%');
                write_serial(*fmt);
            }
        } else {
            write_serial(*fmt);
        }
        fmt++;
    }
    
    va_end(args);
}

// --- Trap Handler ---
void handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t sp) {
    my_printf("\n*** TRAP CAUGHT ***\n");
    my_printf("mcause: "); print_hex(mcause); my_printf("\n");
    my_printf("mepc:   "); print_hex(mepc);   my_printf("\n");
    my_printf("sp:     "); print_hex(sp);     my_printf("\n");
    _exit((int)mcause);
}

// --- Helper Functions ---
static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}

static inline float my_fabs(float x) {
    return x < 0.0f ? -x : x;
}

static int float_equals(float a, float b, float epsilon) {
    return my_fabs(a - b) < epsilon;
}

static int compare_arrays(float* a, float* b, int size, float epsilon) {
    for (int i = 0; i < size; i++) {
        if (!float_equals(a[i], b[i], epsilon)) {
            return 0;
        }
    }
    return 1;
}

static void matmul_reference(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// --- Matmul Vector Implementation ---
static inline void matmul_vector(float* xout, float* x, float* w, int n, int d) {
    my_printf("enter matmul_vector (fp32) with FULL DEBUG:\n");
    static float temp[2048] __attribute__((aligned(64)));
    
    // Clear temp for safety
    for(int k=0; k<2048; k++) temp[k] = 0.0f;

    for (int i = 0; i < d; i++) {
        my_printf("\n--- Row %d ---\n", i);
        
        float* row = w + i * n;
        float* ptr_x = x;
        size_t avl = n;
        size_t temp_idx = 0;
        
        float ref_row_acc = 0.0f; 

        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            
            my_printf("  Chunk: vl=%zu, avl=%zu\n", vl, avl);

            // Load vectors
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row, vl);
            
            // Debug: Verify vrow load
            __riscv_vse32_v_f32m1(&temp[temp_idx], vrow, vl);
            asm volatile("fence rw,rw" ::: "memory");
            debug_delay_cycles(50);
            my_printf("    Loaded w (vec): ");
            for (size_t k = 0; k < vl; k++) {
                my_printf("%f ", temp[temp_idx + k]);
            }
            my_printf("\n");

            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl); 
            
            // Debug: Verify vx load
            __riscv_vse32_v_f32m1(&temp[temp_idx], vx, vl);
            asm volatile("fence rw,rw" ::: "memory");
            debug_delay_cycles(50);
            my_printf("    Loaded x (vec): ");
            for (size_t k = 0; k < vl; k++) {
                my_printf("%f ", temp[temp_idx + k]);
            }
            my_printf("\n");

            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            
            // Store vprod to temp
            __riscv_vse32_v_f32m1(&temp[temp_idx], vprod, vl);
            asm volatile("fence rw,rw" ::: "memory");
            debug_delay_cycles(50);
            
            my_printf("    Vector Prod: ");
            for (size_t k = 0; k < vl; k++) {
                my_printf("%f ", temp[temp_idx + k]);
            }
            my_printf("\n");
            
            // Verify this chunk
            volatile float* vtemp = &temp[temp_idx];
            
            for (size_t k = 0; k < vl; k++) {
                float r_val = row[k];
                float x_val = ptr_x[k];
                float vec_prod = vtemp[k];
                float ref_prod = r_val * x_val;
                
                ref_row_acc += ref_prod;

                my_printf("    Idx %zu: w=%f * x=%f | VecProd=%f | RefProd=%f", temp_idx + k, r_val, x_val, vec_prod, ref_prod);
                
                if (!float_equals(vec_prod, ref_prod, 0.0001f)) {
                     my_printf(" [MISMATCH]\n");
                } else {
                     my_printf(" [OK]\n");
                }
            }

            row += vl;
            ptr_x += vl;
            temp_idx += vl;
            avl -= vl;
        }
        
        // Accumulate vector results
        float acc = 0.0f;
        for (size_t j = 0; j < n; j++) {
            acc += temp[j];
            my_printf("acc%zu=%f\n", j, acc);
        }
        
        xout[i] = acc;
        
        my_printf("  Row Result: VecAcc=%f | RefAcc=%f", acc, ref_row_acc);
        if (!float_equals(acc, ref_row_acc, 0.001f)) {
             my_printf(" [ROW MISMATCH]\n");
        } else {
             my_printf(" [ROW OK]\n");
        }
    }
}

#define N 8
#define D 8

int real_main();

__attribute__((naked)) void main() {
    asm volatile (
        "li t0, 0x200\n"       // MSTATUS_VS mask (bit 9)
        "csrs mstatus, t0\n"   // Set VS bit
        "j real_main\n"        // Jump to real_main
    );
}

int real_main() {
    my_printf("enter real_main:\n");

    // Enable Vectors (VS bit 9 in mstatus) - already done in wrapper, but checking doesn't hurt
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    my_printf("Current mstatus: "); print_hex(mstatus); my_printf("\n");

    float w[D][N];
    float x[N];
    float xout[D];
    float expected[D];
    
    my_printf("initializing data:\n");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < N; j++) {
            w[i][j] = (float)(i + j);
        }
    }
    for (int k = 0; k < N; k++) {
        x[k] = (float)k;
    }

    my_printf("Input Vector x:\n");
    for (int k = 0; k < N; k++) {
        my_printf("%f ", x[k]);
        if ((k + 1) % 8 == 0) my_printf("\n");
    }
    my_printf("\n");

    my_printf("Input Matrix w:\n");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < N; j++) {
            my_printf("%f ", w[i][j]);
        }
        my_printf("\n");
    }

    my_printf("matmul initializing:\n");
    matmul_vector(xout, x, &w[0][0], N, D);

    my_printf("Full xout result:\n");
    for (int i = 0; i < D; i++) {
        my_printf("%f ", xout[i]);
        if ((i + 1) % 8 == 0) my_printf("\n");
    }
    
    my_printf("\nCalculating expected results:\n");
    matmul_reference(expected, x, &w[0][0], N, D);
    
    my_printf("Expected results:\n");
    for (int i = 0; i < D; i++) {
        my_printf("%f ", expected[i]);
        if ((i + 1) % 8 == 0) my_printf("\n");
    }
    
    my_printf("\nComparing results:\n");
    float epsilon = 0.0001f;
    int is_correct = compare_arrays(xout, expected, D, epsilon);
    
    if (is_correct) {
        my_printf("TEST PASSED: Results match expected values within tolerance.\n");
    } else {
        my_printf("TEST FAILED: Results do not match expected values.\n");
        my_printf("Mismatched elements:\n");
        for (int i = 0; i < D; i++) {
            if (!float_equals(xout[i], expected[i], epsilon)) {
                my_printf("Index %d: got %f, expected %f\n", i, xout[i], expected[i]);
            }
        }
    }

    my_printf("\n");
    return 0;
}
