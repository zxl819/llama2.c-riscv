
#include <riscv_vector.h>
#include "uart_helper.c"
#include <string.h>
#include <stdint.h>

// Padding helpers (freestanding-friendly). Must be included before any padded matmul.
#define RVV_PADDING_NO_ALLOCATOR 1
#include "RVV_padding.c"

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

// Debug buffers: place in .bss and force alignment.
// Hardware workaround: on this platform, `vse32` to a fully-aligned base address can drop the first beat.
// Shifting the base pointer by +2 floats (+8 bytes) avoids the failing store path (confirmed by vse_offset_test).
#ifndef RVV_VSE32_WORKAROUND_OFFSET_FLOATS
#define RVV_VSE32_WORKAROUND_OFFSET_FLOATS 4
#endif

#define DBG_ROWS 16
#define DBG_COLS 16
#define DBG_SIZE ((size_t)DBG_ROWS * (size_t)DBG_COLS)

static __attribute__((aligned(64))) float g_debug_vec_a_raw[DBG_SIZE + RVV_VSE32_WORKAROUND_OFFSET_FLOATS];
static __attribute__((aligned(64))) float g_debug_vec_b_raw[DBG_SIZE + RVV_VSE32_WORKAROUND_OFFSET_FLOATS];
static __attribute__((aligned(64))) float g_debug_vec_c_raw[DBG_SIZE + RVV_VSE32_WORKAROUND_OFFSET_FLOATS];
static __attribute__((aligned(64))) float g_debug_vec_d_raw[DBG_SIZE + RVV_VSE32_WORKAROUND_OFFSET_FLOATS];

static float *const g_debug_vec_a = g_debug_vec_a_raw + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;
static float *const g_debug_vec_b = g_debug_vec_b_raw + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;
static float *const g_debug_vec_c = g_debug_vec_c_raw + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;
static float *const g_debug_vec_d = g_debug_vec_d_raw + RVV_VSE32_WORKAROUND_OFFSET_FLOATS;

static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}

static inline void rvv_clear_vstart(void) {
 #if RVV_HAS_VSTART_CSR
    asm volatile("csrw 0x008, zero" ::: "memory");
 #endif
}

// Optional diagnostics for the "lane0/1 keep sentinel" symptom.
// If vstart becomes non-zero (typically after an interrupt/exception during a vector instruction),
// subsequent vector ops may start at lane=vstart and leave earlier lanes untouched.
#ifndef RVV_DEBUG_VSTART
#define RVV_DEBUG_VSTART 1
#endif

// Some implementations (or privilege setups) may not expose the vstart CSR.
// If your hardware traps on `csrr vstart` / `csrw vstart`, set this to 0.
#ifndef RVV_HAS_VSTART_CSR
#define RVV_HAS_VSTART_CSR 0
#endif

#ifndef RVV_FORCE_CLEAR_VSTART
#define RVV_FORCE_CLEAR_VSTART 0
#endif

static inline unsigned long rvv_read_vstart(void) {
 #if RVV_HAS_VSTART_CSR
    unsigned long v;
    asm volatile("csrr %0, 0x008" : "=r"(v) :: "memory");
    return v;
 #else
    return 0;
 #endif
}

static inline void rvv_try_clear_vstart(void) {
#if RVV_HAS_VSTART_CSR
    rvv_clear_vstart();
#endif
}

static inline void rvv_debug_print_vstart_if_nonzero(const char *tag) {
#if RVV_DEBUG_VSTART
    #if RVV_HAS_VSTART_CSR
        unsigned long vstart = rvv_read_vstart();
        if (vstart != 0) {
                print_uart(tag);
                print_uart(" vstart=");
                print_uart_int_dec((int)vstart);
                print_uart("\r\n");
        }
    #else
        // Avoid spamming: print only once.
        static int warned = 0;
        if (!warned) {
                warned = 1;
                print_uart("[DBG] vstart CSR unsupported; cannot read/clear vstart\r\n");
        }
        (void)tag;
    #endif
#else
    (void)tag;
#endif
}

static inline void fill_float_sentinel(float *buf, size_t count, float value) {
    for (size_t i = 0; i < count; i++) {
        buf[i] = value;
    }
}

// Work around a suspected hardware bug where `vse32` may drop the first beat for certain
// destination alignments. We store to a scratch buffer at a known-good offset (+2 floats, +8B)
// and then scalar-copy into the real destination.
// This is intended for DEBUG buffers / correctness experiments, not for high performance.
#ifndef RVV_SAFE_VSE32
#define RVV_SAFE_VSE32 0
#endif

// When RVV_SAFE_VSE32=1, choose a scratch offset (in floats) that avoids the failing store path.
// Empirically on this platform, +3 or +4 tend to work better than +0.
#ifndef RVV_SAFE_VSE32_SCRATCH_OFFSET_FLOATS
#define RVV_SAFE_VSE32_SCRATCH_OFFSET_FLOATS 3
#endif

#if RVV_SAFE_VSE32
static __attribute__((aligned(64))) float g_vse32_scratch[RVV_SAFE_VSE32_SCRATCH_OFFSET_FLOATS + 64];
static inline void safe_vse32_f32m1(float *dst, vfloat32m1_t v, size_t vl) {
    // Store to scratch+OFFSET then copy back.
    float *tmp = g_vse32_scratch;
    __riscv_vse32_v_f32m1(tmp + RVV_SAFE_VSE32_SCRATCH_OFFSET_FLOATS, v, vl);
    //asm volatile("fence rw, rw" ::: "memory");
    debug_delay_cycles(100);
    for (size_t i = 0; i < vl; i++) dst[i] = tmp[RVV_SAFE_VSE32_SCRATCH_OFFSET_FLOATS + i];
}
#else
static inline void safe_vse32_f32m1(float *dst, vfloat32m1_t v, size_t vl) {
    __riscv_vse32_v_f32m1(dst, v, vl);
    debug_delay_cycles(100);
}
#endif

// Disable machine interrupts (MSTATUS.MIE) and return the old mstatus.
// Use irq_restore(old) to restore MIE to the previous state.
static inline unsigned long irq_disable_save(void) {
    unsigned long old;
    asm volatile("csrrc %0, mstatus, %1" : "=r"(old) : "r"(0x8UL) : "memory");
    return old;
}

static inline void irq_restore(unsigned long old_mstatus) {
    if (old_mstatus & 0x8UL) {
        asm volatile("csrs mstatus, %0" :: "r"(0x8UL) : "memory");
    } else {
        asm volatile("csrc mstatus, %0" :: "r"(0x8UL) : "memory");
    }
}

typedef long ssize_t;

// Override the weak handle_trap in crt.S so traps don't silently hang/exit.
// This prints key CSRs over UART and then halts.
uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t sp) {
    (void)sp;
    unsigned long mtval = 0;
    unsigned long mstatus = 0;
    asm volatile("csrr %0, mtval" : "=r"(mtval));
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));

    print_uart("\r\n[TRAP] mcause=0x");
    print_uart_hex((unsigned long long)mcause);
    print_uart(" mepc=0x");
    print_uart_hex((unsigned long long)mepc);
    print_uart(" mtval=0x");
    print_uart_hex((unsigned long long)mtval);
    print_uart(" mstatus=0x");
    print_uart_hex((unsigned long long)mstatus);
    print_uart("\r\n");

    while (1) {
        asm volatile("wfi");
    }

    return mepc;
}

// ---------------------------------------------------------------------------
// Tiny bump allocator for bare-metal use
#ifndef BARE_HEAP_BYTES
#define BARE_HEAP_BYTES (32 * 1024 * 1024)
#endif

// Default allocation alignment for the bare-metal bump allocator.
// Keep this in one place to avoid hard-coded alignment values across files.
#ifndef BARE_ALLOC_ALIGN
#define BARE_ALLOC_ALIGN 64
#endif
static unsigned char bare_heap[BARE_HEAP_BYTES];
static size_t bare_heap_offset = 0;

static void panic(const char *msg) {
    print_uart("[bare] PANIC: ");
    print_uart(msg);
    print_uart("\n");
}

static void *bare_alloc(size_t size, size_t alignment) {
    if (alignment == 0) alignment = BARE_ALLOC_ALIGN;
    size_t misalignment = bare_heap_offset % alignment;
    if (misalignment != 0) bare_heap_offset += alignment - misalignment;
    if (bare_heap_offset + size > BARE_HEAP_BYTES) panic("bare heap exhausted");
    void *ptr = &bare_heap[bare_heap_offset];
    bare_heap_offset += size;
    return ptr;
}

void *malloc(size_t size) {
    if (size == 0) size = 1;
    return bare_alloc(size, BARE_ALLOC_ALIGN);
}

void *calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return bare_alloc(1, BARE_ALLOC_ALIGN);
    size_t total = count * size;
    void *ptr = bare_alloc(total, BARE_ALLOC_ALIGN);
    // Zero out the memory
    unsigned char *p = (unsigned char *)ptr;
    for (size_t i = 0; i < total; i++) {
        p[i] = 0;
    }
    return ptr;
}

void free(void *ptr) { (void)ptr; }


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

static void __attribute__((noinline)) matmul(float* xout, float* x, float* w, int n, int d) {
    // Requirement: ensure input x length (n) and each W row length are multiples of 16.
    // For RVV e32m1 with VLEN=512, vlmax is 16; using fixed vl=16 avoids tail handling.
    if ((n & 15) != 0) {
        print_uart("Error: matmul requires n % 16 == 0\r\n");
        return;
    }
    if (n <= 0 || d <= 0) return;

    const size_t vl = __riscv_vsetvl_e32m1(16); // fixed 16-lane chunks
    const vfloat32m1_t vzero = __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vmv_v_x_u32m1(0u, vl));
    for (int i = 0; i < d; i++) {
        float* row_ptr = w + (size_t)i * (size_t)n;
        float* x_ptr = x;

        float acc = 0.0f;
        // n is guaranteed multiple of 16, so this loop has no tail.
        for (int j = 0; j < n; j += 16) {
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row_ptr, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_ptr, vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);

            // Fallback reduction: store to a local buffer and sum scalars.
            __attribute__((aligned(64))) float lane_buf[16];
            __riscv_vse32_v_f32m1(lane_buf, vprod, vl);
            debug_delay_cycles(100);
            for (size_t k = 0; k < vl; k++) acc += lane_buf[k];

            row_ptr += 16;
            x_ptr += 16;
        }

        xout[i] = acc;
        debug_delay_cycles(1000);
    }
}

static void __attribute__((noinline)) matmul_vector2(float* xout, float* x, float* w, int n, int d) {

    // --- 1. Padding 准备 ---
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vzero = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(0u, vlmax));
    // 计算大于等于 n 的最小的 vlmax 的倍数
    size_t padded_n = ((n + vlmax - 1) / vlmax) * vlmax;

    // --- 2. 使用通用函数进行 Padding ---
    // 为向量 x 进行 padding
    PaddedArray1D padded_x = pad_array_1d(x, n, sizeof(float), padded_n);
    if (!padded_x.data) {
        print_uart("Error: Failed to pad vector x!\r\n");
        return;
    }
    // 为矩阵 w 进行 padding
    PaddedArray2D padded_w = pad_matrix_2d(w, d, n, sizeof(float), padded_n);
    if (!padded_w.data) {
        print_uart("Error: Failed to pad matrix w!\r\n");
        free_padded_array_1d(&padded_x); // 释放已分配的内存
        return;
    }
    
    // --- 3. 核心计算 (使用填充后的数据) ---
    for (int i = 0; i < d; i++) {
        float* row_ptr = (float*)padded_w.data + (size_t)i * padded_w.padded_cols;
        float* x_ptr = (float*)padded_x.data;

        // 用 1-lane 向量保存该行的累加和，避免落回标量循环
        size_t vl1 = __riscv_vsetvl_e32m1(16);
        vfloat32m1_t vacc1 = vzero;

        size_t remaining = padded_n;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row_ptr, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_ptr, vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);

            // 归约本块求和（结果在 lane0），并累加到 vacc1
            vfloat32m1_t vblk = __riscv_vfredosum_vs_f32m1_f32m1(vprod, vzero, vl);
            vacc1 = __riscv_vfadd_vv_f32m1(vacc1, vblk, vl1);

            row_ptr += vl;
            x_ptr += vl;
            remaining -= vl;
        }

        // 写出该行结果：xout[i] = sum_j w[i,j]*x[j]
        __riscv_vse32_v_f32m1(&xout[i], vacc1, vl1);
        debug_delay_cycles(100);
    }

    // free_padded_matrix_2d(&padded_w);
    // free_padded_array_1d(&padded_x);

}

/**
 * @brief 使用向量指令实现矩阵-向量乘法 (fp32)
 *
 * @param xout 输出向量 (d x 1)
 * @param x    输入向量 (n x 1)
 * @param w    输入矩阵 (d x n)
 * @param n    向量x的维度和矩阵w的列数
 * @param d    矩阵w的行数和输出向量xout的维度
 */
static void __attribute__((noinline)) matmul_vector(float* xout, float* x, float* w, int n, int d,float * debug_vec_d) {
    print_uart("enter matmul_vector (fp32) with FULL DEBUG:\r\n");
    for (int i = 0; i < d; i++) {
        float* row = w + i * n;
        float* ptr_x = x;
        size_t avl = n;
        size_t block_offset_in_row = 0; 
        
        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);

            // Load vectors
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);

            // Store vprod to temp
            float* dest_ptr_d = &debug_vec_d[i * n + block_offset_in_row];
            __riscv_vse32_v_f32m1(dest_ptr_d, vprod, vl);
            debug_delay_cycles(100);

            row += vl;
            ptr_x += vl;
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
        }
        xout[i] = acc;
}
}

// (RVV_padding.c already included at file top)
/**
 * @brief 使用向量指令和内部padding策略实现矩阵-向量乘法
 *
 * 此函数会自动将输入的x和w填充到VLEN的整数倍长度，以优化循环。
 * 
 *
 * @param xout 输出向量
 * @param x    输入向量
 * @param w    输入矩阵
 * @param n    向量x的维度和矩阵w的列数
 * @param d    矩阵w的行数和输出向量xout的维度
 * @param debug_vec_d 用于存储中间乘积结果的调试缓冲区 (大小为 d * n)
 */
static void __attribute__((noinline)) matmul_vector_padded(float* xout, float* x, float* w, int n, int d, float * debug_vec_a, float * debug_vec_b,float * debug_vec_c, float * debug_vec_d) {
    print_uart("enter matmul_vector (fp32) with INTERNAL PADDING:\r\n");

    // If vstart is non-zero (e.g., left over after an exception), RVV loads/stores may start from lane vstart,
    // leaving the first lanes untouched (often showing up as 0.0 after memset). Clear it proactively.
    //rvv_clear_vstart();

    // --- 1. Padding 准备 ---
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    // 计算大于等于 n 的最小的 vlmax 的倍数
    size_t padded_n = ((n + vlmax - 1) / vlmax) * vlmax;

    // --- 2. 使用通用函数进行 Padding ---
    // 为向量 x 进行 padding
    PaddedArray1D padded_x = pad_array_1d(x, n, sizeof(float), padded_n);
    if (!padded_x.data) {
        print_uart("Error: Failed to pad vector x!\r\n");
        return;
    }
    float *padded_x_f = (float *)padded_x.data;
    print_uart("Input Vector x after padding:\r\n");
    for (size_t k = 0; k < padded_x.padded_len; k++) {
        print_uart_float(padded_x_f[k]);
        print_uart(" ");
        if (((k + 1) % 8) == 0) print_uart("\r\n");
    }
    if ((padded_x.padded_len % 8) != 0) print_uart("\r\n");
    


    // 为矩阵 w 进行 padding
    PaddedArray2D padded_w = pad_matrix_2d(w, d, n, sizeof(float), padded_n);
    if (!padded_w.data) {
        print_uart("Error: Failed to pad matrix w!\r\n");
        free_padded_array_1d(&padded_x); // 释放已分配的内存
        return;
    }
    float *padded_w_f = (float *)padded_w.data;
    print_uart("\r\nInput Matrix w after padding:\r\n");
    for (int i = 0; i < d; i++) {
        for (size_t j = 0; j < padded_w.padded_cols; j++) {
            print_uart_float(padded_w_f[(size_t)i * padded_w.padded_cols + j]);
            print_uart(" ");
            if (((j + 1) % 8) == 0) print_uart("\r\n");
        }
        if ((padded_w.padded_cols % 8) != 0) print_uart("\r\n");
        print_uart("\r\n");
    }
    

    print_uart("vlmax=");
    print_uart_int_dec((int)vlmax);
    print_uart(" padded_n=");
    print_uart_int_dec((int)padded_n);
    print_uart("\r\n");

    // --- 3. 核心计算 (使用填充后的数据) ---
    for (int i = 0; i < d; i++) {
        //rvv_clear_vstart();
        size_t block_offset_in_row = 0;
        // 指向填充后矩阵的第 i 行
        float* row_padded = (float*)padded_w.data + (size_t)i * padded_w.padded_cols;
        float* ptr_x_padded = (float*)padded_x.data;
        float ref_row_acc = 0.0f; 

        size_t j = 0;
        size_t remaining = padded_n;
        while (remaining > 0) {
            //rvv_clear_vstart();
            rvv_debug_print_vstart_if_nonzero("[DBG] before padded block");
#if RVV_FORCE_CLEAR_VSTART
            rvv_clear_vstart();
#endif
            // 当前块的标量指针（用于 ref 校验）
            float* row_block = row_padded;
            float* x_block = ptr_x_padded;

            // Wrap the RVV critical window (load/mul/store) with interrupts disabled.
            unsigned long irq_state = irq_disable_save();
            size_t vl = __riscv_vsetvl_e32m1(remaining);

            // Load
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row_padded, vl);
            float* dest_ptr = &debug_vec_a[(size_t)i * padded_n + block_offset_in_row];
            safe_vse32_f32m1(dest_ptr, vrow, vl);

            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x_padded, vl);
            float* dest_ptr_b = &debug_vec_b[(size_t)i * padded_n + block_offset_in_row];
            safe_vse32_f32m1(dest_ptr_b, vx, vl);
            debug_delay_cycles(50000);

            // Mul
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            float* dest_ptr_c = &debug_vec_c[(size_t)i * padded_n + block_offset_in_row];
            safe_vse32_f32m1(dest_ptr_c, vprod, vl);
            debug_delay_cycles(50000);

            // Store vprod to the debug buffer (stride = padded_n)
            float* dest_ptr_d = &debug_vec_d[(size_t)i * padded_n + block_offset_in_row];
            safe_vse32_f32m1(dest_ptr_d, vprod, vl);
            debug_delay_cycles(50000);
            irq_restore(irq_state);

            print_uart("DEBUG: About to vle32.v from row ptr: 0x");
            print_uart_hex((unsigned long long)row_padded);
            print_uart("\r\n");

            //debug_delay_cycles(50000);
            print_uart("    Loaded w (vec): ");
            for (size_t k = 0; k < vl; k++) {
                print_uart_float(dest_ptr[k]);
                print_uart(" ");
            }
            print_uart("\r\n");

            // Probe store->scalar visibility: re-read lane0/1 after a short delay
            debug_delay_cycles(5000);
            print_uart("    ReRead w[0..1]: ");
            if (vl > 0) {
                print_uart_float(dest_ptr[0]);
                print_uart(" ");
            }
            if (vl > 1) {
                print_uart_float(dest_ptr[1]);
            }
            print_uart("\r\n");

            debug_delay_cycles(50000);
            print_uart("    Loaded x: ");
            for (size_t k = 0; k < vl; k++) {
                print_uart_float(dest_ptr_b[k]);
                print_uart(" ");
            }
            print_uart("\r\n");

            debug_delay_cycles(5000);
            print_uart("    ReRead x[0..1]: ");
            if (vl > 0) {
                print_uart_float(dest_ptr_b[0]);
                print_uart(" ");
            }
            if (vl > 1) {
                print_uart_float(dest_ptr_b[1]);
            }
            print_uart("\r\n");

            debug_delay_cycles(50000);
            print_uart("    Vector Prod: ");
            for (size_t k = 0; k < vl; k++) {
                print_uart_float(dest_ptr_c[k]);
                print_uart(" ");
            }
            print_uart("\r\n");

            debug_delay_cycles(5000);
            print_uart("    ReRead prod[0..1]: ");
            if (vl > 0) {
                print_uart_float(dest_ptr_c[0]);
                print_uart(" ");
            }
            if (vl > 1) {
                print_uart_float(dest_ptr_c[1]);
            }
            print_uart("\r\n");

            debug_delay_cycles(50000);
            for (size_t k = 0; k < vl; k++) {
                size_t idx = j + k;
                float r_val = row_block[k];
                float x_val = x_block[k];
                float vec_prod = dest_ptr_c[k]; //error here
                float ref_prod = r_val * x_val;
                if (idx < (size_t)n) ref_row_acc += ref_prod;

                print_uart("    Idx "); print_uart_int_dec((int)idx);
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

            row_padded += vl;
            ptr_x_padded += vl;
            block_offset_in_row += vl;

            j += vl;
            remaining -= vl;
        }
        
        // Accumulate vector results
        float acc = 0.0f;
        // 计算当前行在 debug_vec_d 中的起始地址（stride = padded_n）
        float* current_row_products = &debug_vec_c[(size_t)i * padded_n];
        // print_uart("  Accumulating temp array:\r\n");
        for (size_t j = 0; j < (size_t)n; j++) {
            acc += current_row_products[j];
            // print_uart("acc");
            // print_uart_int_dec(j);
            // print_uart("=");
            // print_uart_float(acc);
            // print_uart("\r\n");
            // print_uart(" ");
            // if (((j + 1) % 16) == 0) print_uart("\r\n");
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



    // // --- 4. 将填充后的调试数据复制回原始缓冲区 ---
    // // 这确保调用者看到的 debug_vec_d 是基于原始维度 n 的
    // for (int i = 0; i < d; i++) {
    //     memcpy(&debug_vec_d[i * n], (float*)debug_vec_d_padded.data + (size_t)i * padded_n, (size_t)n * sizeof(float));
    // }
    // float *debug_vec_d_f = (float *)debug_vec_d;
    // print_uart("debug_vec_d:\r\n");
    // for (int i = 0; i < d; i++) {
    //     for (int j = 0; j < n; j++) {
    //         print_uart_float(debug_vec_d_f[(size_t)i * (size_t)n + (size_t)j]);
    //         print_uart(" ");
    //         if (((j + 1) % 8) == 0) {
    //             print_uart("\r\n");
    //         }
    //     }

    //     print_uart("\r\n");
    // }
    // debug_delay_cycles(5000);

    // --- 4. 释放临时内存 ---
    //free_padded_array_1d(&padded_x);
    //free_padded_matrix_2d(&padded_w);
}

static void __attribute__((noinline)) matmul_vector_padded_withoutdebug(float* xout, float* x, float* w, int n, int d,float * debug_vec_c) {
    //print_uart("enter matmul_vector (fp32) with INTERNAL PADDING:\r\n");

    // If vstart is non-zero (e.g., left over after an exception), RVV loads/stores may start from lane vstart,
    // leaving the first lanes untouched (often showing up as 0.0 after memset). Clear it proactively.
    //rvv_clear_vstart();

    // --- 1. Padding 准备 ---
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    // 计算大于等于 n 的最小的 vlmax 的倍数
    size_t padded_n = ((n + vlmax - 1) / vlmax) * vlmax;

    // --- 2. 使用通用函数进行 Padding ---
    // 为向量 x 进行 padding
    PaddedArray1D padded_x = pad_array_1d(x, n, sizeof(float), padded_n);
    if (!padded_x.data) {
        print_uart("Error: Failed to pad vector x!\r\n");
        return;
    }
    // 为矩阵 w 进行 padding
    PaddedArray2D padded_w = pad_matrix_2d(w, d, n, sizeof(float), padded_n);
    if (!padded_w.data) {
        print_uart("Error: Failed to pad matrix w!\r\n");
        free_padded_array_1d(&padded_x); // 释放已分配的内存
        return;
    }
    
    // --- 3. 核心计算 (使用填充后的数据) ---
    for (int i = 0; i < d; i++) {
        //rvv_clear_vstart();
        size_t block_offset_in_row = 0;
        // 指向填充后矩阵的第 i 行
        float* row_padded = (float*)padded_w.data + (size_t)i * padded_w.padded_cols;
        float* ptr_x_padded = (float*)padded_x.data;
        float ref_row_acc = 0.0f; 

        size_t j = 0;
        size_t remaining = padded_n;
        while (remaining > 0) {
            float* row_block = row_padded;
            float* x_block = ptr_x_padded;

            // Wrap the RVV critical window (load/mul/store) with interrupts disabled.
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            // Load
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row_padded, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x_padded, vl);
            // Mul
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            float* dest_ptr_c = &debug_vec_c[(size_t)i * padded_n + block_offset_in_row];
            safe_vse32_f32m1(dest_ptr_c, vprod, vl);
          
            row_padded += vl;
            ptr_x_padded += vl;
            block_offset_in_row += vl;

            j += vl;
            remaining -= vl;
        }
        
        // Accumulate vector results
        float acc = 0.0f;
        // 计算当前行在 debug_vec_d 中的起始地址（stride = padded_n）
        float* current_row_products = &debug_vec_c[(size_t)i * padded_n];
        // print_uart("  Accumulating temp array:\r\n");
        for (size_t j = 0; j < (size_t)n; j++) {
            acc += current_row_products[j];
        }
        xout[i] = acc;
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
    __attribute__((aligned(64))) float w[d][n];
    // 2. 正确分配 x 为一个长度为 n 的向量
    __attribute__((aligned(64))) float x[n];
    // 3. 为结果分配 xout，长度为 d
    __attribute__((aligned(64))) float xout[d];
     // 4. 为参考结果分配 expected，长度为 d
    __attribute__((aligned(64))) float expected[d];

    // Use printable float sentinels (memset(1/2/3/4) creates tiny denormals that often print as 0.0)
    fill_float_sentinel(g_debug_vec_a, DBG_SIZE, 1111.0f);
    fill_float_sentinel(g_debug_vec_b, DBG_SIZE, 2222.0f);
    fill_float_sentinel(g_debug_vec_c, DBG_SIZE, 3333.0f);
    fill_float_sentinel(g_debug_vec_d, DBG_SIZE, 4444.0f);
    print_uart("initializing data:\r\n");
    // 初始化 w 和 x
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            w[i][j] = (float)(i + j + 1); // 给一个示例值
        }
    }
    for (int k = 0; k < n; k++) {
        x[k] = (float)k + 1;
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
    //matmul_vector_padded_withoutdebug(xout, x, &w[0][0], n, d,g_debug_vec_d);
    matmul(xout, x, &w[0][0], n, d);
    //matmul_vector(xout, x, &w[0][0], n, d,g_debug_vec_d);
   // matmul_vector_padded(xout, x, &w[0][0], n, d, g_debug_vec_a, g_debug_vec_b, g_debug_vec_c, g_debug_vec_d);
    //matmul_vector_padded(xout, x, &w[0][0], n, d, &debug_vec_d[0][0]);

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