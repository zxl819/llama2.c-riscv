#ifndef RISCV_VECTOR_KERNELS_H
#define RISCV_VECTOR_KERNELS_H

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include "RVV_padding.c"
//#include "uart_helper.c"

// These UART helpers are provided by the including translation unit in bare-metal builds.
// They must be visible before this header uses them (C99 forbids implicit declarations).
void print_uart(const char *str);
void print_uart_float(float val);

static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}

static inline void safe_vse32_f32m1(float *dst, vfloat32m1_t v, size_t vl) {
    __riscv_vse32_v_f32m1(dst, v, vl);
    debug_delay_cycles(100);
}
// ---------------------------------------------------------------------------

static inline void softmax_vec(float* x, int size) {
    // init_uart(CLOCK_FREQUENCY, UART_BITRATE); // Removing repeated init which might cause hangs
    //print_uart("Enter the softmax.\r\n");
    float* dst = x;
    float* src = x;
    size_t n = size;
    
    const size_t vlmax = __riscv_vsetvlmax_e32m1(); 
    
    // 使用内存数组存储所有常量，通过整数指针访问
    static const uint32_t constants[] = {
        0xFF800000,  // -INFINITY [0]
        0x3F317218,  // ln(2)     [1]
        0x3FB8AA3B,  // 1/ln(2)   [2]  
        0x3E800000,  // 0.25f     [3]
        0x40000000,  // 2.0f      [4]
        0x3F800000,  // 1.0f      [5]
        0x3EFFFFFC,  // poly_c_2  [6]
        0x3D2AA427,  // poly_c_3  [7]
        0x3AAAB664,  // poly_c_4  [8]
        0x38891F99,  // poly_c_5  [9]
        0x3684250E,  // poly_c_6  [10]
        0x322BCC77   // 1e-8f     [11]
    };
    
    // 常量（按位）
    const uint32_t BITS_NEG_INF = 0xFF800000u;
    const uint32_t BITS_ILN2    = 0x3FB8AA3Bu; // 1/ln(2)
    const uint32_t BITS_LN2_HI  = 0x3F317000u; // ln2 hi
    const uint32_t BITS_LN2_LO  = 0x38800C00u; // ln2 lo
    const uint32_t BITS_EPS     = 0x322BCC77u; // 1e-8
    const uint32_t BITS_TWO     = 0x40000000u; // 2.0

    // e^r 的 8 阶多项式系数（Horner）
    const uint32_t C0 = 0x3F800000u; // 1
    const uint32_t C1 = 0x3F800000u; // 1
    const uint32_t C2 = 0x3F000000u; // 1/2
    const uint32_t C3 = 0x3E2AAAABu; // 1/6
    const uint32_t C4 = 0x3D2AAAABu; // 1/24
    const uint32_t C5 = 0x3C088889u; // 1/120
    const uint32_t C6 = 0x3AB60B61u; // 1/720
    const uint32_t C7 = 0x3A1175D4u; // 1/5040
    const uint32_t C8 = 0x3926ED8Eu; // 1/40320

    // 预构建常量向量（用 vlmax，后续算子用当前 vl，只会影响前 vl 个元素）
    vfloat32m1_t vzero   = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(0u,        vlmax));
    vfloat32m1_t vnegInf = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_NEG_INF, vlmax));
    vfloat32m1_t viln2   = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_ILN2,   vlmax));
    vfloat32m1_t vln2_hi = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_LN2_HI, vlmax));
    vfloat32m1_t vln2_lo = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_LN2_LO, vlmax));
    vfloat32m1_t pc0 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C0, vlmax));
    vfloat32m1_t pc1 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C1, vlmax));
    vfloat32m1_t pc2 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C2, vlmax));
    vfloat32m1_t pc3 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C3, vlmax));
    vfloat32m1_t pc4 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C4, vlmax));
    vfloat32m1_t pc5 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C5, vlmax));
    vfloat32m1_t pc6 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C6, vlmax));
    vfloat32m1_t pc7 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C7, vlmax));
    vfloat32m1_t pc8 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C8, vlmax));
    vfloat32m1_t vtwo1 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_TWO,    vlmax));
    vfloat32m1_t veps1 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_EPS,    vlmax));

    
    vuint32m1_t vneg_inf_int = __riscv_vmv_v_x_u32m1(constants[0], vlmax);
    vfloat32m1_t vmax = __riscv_vreinterpret_v_u32m1_f32m1(vneg_inf_int);
    
    // ===============Pass-1: 求全局最大值 max_x====================================
    float* src_orig = src; // 保存原始指针
    size_t avl = n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
        vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
        avl -= vl;
        src += vl;
    }
    src = src_orig; // 重置源指针
    // reduce -> 单元素向量，元素0为 max_x
    vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m1_f32m1(vmax, vnegInf, vlmax);


    // final maximum reduction
    vfloat32m1_t vredmax = __riscv_vreinterpret_v_u32m1_f32m1(vneg_inf_int);
    vredmax = __riscv_vfredmax_vs_f32m1_f32m1(vmax, vredmax, vlmax);

    // 通过内存获取 max_x，避免标量浮点寄存器
    uint32_t max_x_bits[32] __attribute__((aligned(64))); // Increased size for safety (VLEN coverage)

    vuint32m1_t vredmax_int = __riscv_vreinterpret_v_f32m1_u32m1(vredmax);
    __riscv_vse32_v_u32m1(max_x_bits, vredmax_int, 1);

    union { uint32_t u; float f; } max_x_union = {.u = max_x_bits[0]};
    uint32_t max_x_as_int = max_x_union.u;

    //=================== Pass-2: 计算 exp(x - max_x)，写入 dst，并累加求和（和保存在单元素向量 vsum1 中）========================
    vfloat32m1_t vsum1 = vzero;
    float* dst_orig = dst;
    size_t avl2 = n;
    while (avl2 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl2);
        size_t pos = n - avl2; // 本块起始偏移
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);

        // 广播 max_x 到本块长度（不经标量浮点）
        vfloat32m1_t vmax_x = __riscv_vrgather_vx_f32m1(vmax1, 0, vl);
        vx = __riscv_vfsub_vv_f32m1(vx, vmax_x, vl); // x - max_x
        // k = round((x-max)/ln2)
        unsigned old_frm;
        asm volatile("csrr %0, frm" : "=r"(old_frm));
        asm volatile("csrw frm, %0" :: "r"(0)); // RNE
        vfloat32m1_t vxiln2 = __riscv_vfmul_vv_f32m1(vx, viln2, vl);
        vint32m1_t   vk     = __riscv_vfcvt_x_f_v_i32m1(vxiln2, vl);
        asm volatile("csrw frm, %0" :: "r"(old_frm));

        // r = (x-max) - k*ln2_hi - k*ln2_lo
        vfloat32m1_t vfk    = __riscv_vfcvt_f_x_v_f32m1(vk, vl);
        vfloat32m1_t vkl2hi = __riscv_vfmul_vv_f32m1(vfk, vln2_hi, vl);
        vfloat32m1_t vkl2lo = __riscv_vfmul_vv_f32m1(vfk, vln2_lo, vl);
        vfloat32m1_t vr     = __riscv_vfsub_vv_f32m1(vx, vkl2hi, vl);
        vr = __riscv_vfsub_vv_f32m1(vr, vkl2lo, vl);

        // p = poly_e(vr)（8 阶 Horner）
        vfloat32m1_t p = pc8;

        p = __riscv_vfmadd_vv_f32m1(p, vr, pc7, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc6, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc5, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc4, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc3, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc2, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc1, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc0, vl);
        
       

        //2^k 重建：((k+127)<<23)
        const int exp_bias = 127;
        vint32m1_t vbiased = __riscv_vadd_vx_i32m1(vk, exp_bias, vl);
        vint32m1_t vexp2   = __riscv_vsll_vx_i32m1(vbiased, 23, vl);
        vfloat32m1_t vf2k  = __riscv_vreinterpret_v_i32m1_f32m1(vexp2);

        //exp = p * 2^k
        vfloat32m1_t vexp  = __riscv_vfmul_vv_f32m1(p, vf2k, vl);
        // 写出 exp，并累加块和 -> 总和保存在单元素向量 vsum1
        __riscv_vse32_v_f32m1(dst, vexp, vl);
        vfloat32m1_t vblk  = __riscv_vfredosum_vs_f32m1_f32m1(vexp, vzero, vl);
        vsum1 = __riscv_vfadd_vv_f32m1(vsum1, vblk, __riscv_vsetvl_e32m1(1));

        avl2 -= vl;
        src  += vl;
        dst  += vl;
    }

    // 计算 inv(sum)（单元素向量），避免标量浮点
    size_t vl1 = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vsum1_eps = __riscv_vfadd_vv_f32m1(vsum1, veps1, vl1);
    vfloat32m1_t vinv1 = __riscv_vfrec7_v_f32m1(vsum1_eps, vl1);

    // NR refine ×2：vinv = vinv * (2 - d*vinv)
    vfloat32m1_t corr = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1); // 2 - d*x
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    corr  = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1);
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    // 1. 先把单元素倒数 vinv1 写到 inv_table
    uint32_t inv_table[32] __attribute__((aligned(64))); // Increased size for safety

    __riscv_vse32_v_u32m1(inv_table, __riscv_vreinterpret_v_f32m1_u32m1(vinv1), 1);


    // 2. 构造单元素向量
    vfloat32m1_t inv_vec = __riscv_vreinterpret_v_u32m1_f32m1(
    __riscv_vle32_v_u32m1(inv_table, 1));

    // =================Pass-3: 归一化（用 vrgather_vx 将单元素 vinv1 广播到当前 vl）=========================
    dst = dst_orig;
    size_t avl3 = n;
    while (avl3 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl3);
        size_t pos = n - avl3;
        vfloat32m1_t row   = __riscv_vle32_v_f32m1(dst, vl);
        // 用 vrgather.vi 广播 inv_vec[0] 到所有 lane
        vfloat32m1_t vinvB = __riscv_vrgather_vx_f32m1(inv_vec, 0, vl); //这里编译器会变为vrgather.vi
        // 归一化
        row = __riscv_vfmul_vv_f32m1(row, vinvB, vl);
        __riscv_vse32_v_f32m1(dst, row, vl);

        avl3 -= vl;
        dst  += vl;
    }
    debug_delay_cycles(500000);
    //print_uart("End the softmax. \r\n");
    // if (canary1 != 0xDEADBEEF) print_uart("PANIC: Stack Corruption (canary1 at exit)!\r\n");
    // if (canary2 != 0xCAFEBABE) print_uart("PANIC: Stack Corruption (canary2 at exit)!\r\n");
    // if (canary3 != 0xBAADF00D) print_uart("PANIC: Stack Corruption (canary3 at exit)!\r\n");
    // if (canary4 != 0xFEEDFACE) print_uart("PANIC: Stack Corruption (canary4 at exit)!\r\n");
    // print_uart("Softmax returning...\r\n");
}

static inline void element_add(float* x, float* y, int size) {
    size_t avl = size;
    float* ptr_x = x;
    float* ptr_y = y;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(ptr_y, vl);
        vfloat32m1_t vres = __riscv_vfadd_vv_f32m1(vx, vy, vl);
        __riscv_vse32_v_f32m1(ptr_x, vres, vl);
        ptr_x += vl;
        ptr_y += vl;
        avl -= vl;
    }
}
static void __attribute__((noinline)) matmul(float* xout, float* x, float* w, int n, int d) {
    // Requirement: ensure input x length (n) and each W row length are multiples of 16.
    // For RVV e32m1 with VLEN=512, vlmax is 16; using fixed vl=16 avoids tail handling.
    if ((n & 15) != 0) {
        print_uart("Error: matmul requires n % 16 == 0\r\n");
        return;
    }
    if (n <= 0 || d <= 0) return;

    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const vfloat32m1_t vzero = __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vmv_v_x_u32m1(0u, vlmax));
    const size_t vl1 = __riscv_vsetvl_e32m1(16);

    for (int i = 0; i < d; i++) {
        float* row_ptr = w + (size_t)i * (size_t)n;
        float* x_ptr = x;

        // Accumulate chunk sums in a 1-lane vector accumulator.
        vfloat32m1_t vacc1 = vzero;

        // n is guaranteed multiple of 16, so this loop has no tail.
        for (int j = 0; j < n; j += 16) {
            const size_t vl = __riscv_vsetvl_e32m1(16);
            const vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row_ptr, vl);
            const vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_ptr, vl);
            const vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            const vfloat32m1_t vblk = __riscv_vfredosum_vs_f32m1_f32m1(vprod, vzero, vl);
            vacc1 = __riscv_vfadd_vv_f32m1(vacc1, vblk, vl1);
            row_ptr += 16;
            x_ptr += 16;
        }

        __riscv_vse32_v_f32m1(&xout[i], vacc1, vl1);
        debug_delay_cycles(10);
    }
}
static inline void matmul_vector(float* xout, float* x, float* w, int n, int d) {
    static float debug_capture[2048] __attribute__((aligned(64)));
    static int debug_captured = 0;
    static int capture_idx = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    //vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vzero   = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(0u,        vlmax));
    
    // // 移到循环外，移除过度的对齐要求，避免栈指针动态调整带来的风险
    static float temp[2048] __attribute__((aligned(64))); 
    float temp2[32] __attribute__((aligned(64))); 


    for (int i = 0; i < d; i++) {
        float* row = w + i * n;
        float* ptr_x = x;
        
        vfloat32m1_t vsum = vzero;
        size_t avl = n;
        float acc = 0.0f;
        
        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(row, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(ptr_x, vl);

            // Capture debug data safely (all chunks of first row)
            if (i == 0 && !debug_captured) {
                if (capture_idx + vl <= 2048) {
                    __riscv_vse32_v_f32m1(&debug_capture[capture_idx], vx, vl);
                    debug_delay_cycles(5);
                    capture_idx += vl;
                }
            }

            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vx, vl);
            
            vsum = __riscv_vfadd_vv_f32m1(vsum, vprod, vl);
            // 创建掩码向量（全为1，表示更新所有元素）
            // 使用掩码进行向量加法
            //vsum = __riscv_vfadd_vv_f32m1_tu(mask, vsum, vprod, vl);
            //vsum = __riscv_vfadd_vv_f32m1_tu(vsum, vsum, vprod, vl);

            row += vl;
            ptr_x += vl;
            avl -= vl;
        }
        
        if (i == 0 && !debug_captured) {
            debug_captured = 1; // Mark capture finished
        }
        
        // // 重置 vl 为 vlmax 以确保保存所有累加结果
        size_t vl_store = __riscv_vsetvl_e32m1(vlmax);
        
        // // 使用栈上的临时数组暂存向量结果
        // // 注意：不要使用 = {0} 初始化，这会导致隐式 memset 调用，可能破坏寄存器状态或导致栈异常
        // //float temp[32] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(temp, vsum, vl_store);
        
        // // 关键修复：添加内存屏障，确保向量存储对标量加载可见
        asm volatile("fence rw,rw" ::: "memory");
        
        debug_delay_cycles(50);
        //float acc = 0.0f;
        volatile float* vtemp = temp;
        for (size_t j = 0; j < vl_store; j++) {
            acc += vtemp[j];
            //temp[j]=0; // Don't clear static temp, just overwrite next time
        }
        
        xout[i] = acc;
    }

    if (debug_captured == 1) {
        print_uart("Captured Vector Input (Full): ");
        print_uart("\r\n");
        for(int k=0; k<capture_idx; k++) {
             print_uart_float(debug_capture[k]);
             print_uart(" ");
             if ((k + 1) % 7 == 0) print_uart("\r\n");
        }
        print_uart("\r\n");
        debug_captured = 2; // Mark as printed
    }
    //debug_delay_cycles(5000);
}


static void __attribute__((noinline)) matmul_vector2(float* xout, float* x, float* w, int n, int d) {  
    // Keep an alternate entry point, but make it allocation-free as well.
    matmul(xout, x, w, n, d);
}


#endif
