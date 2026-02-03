/*
============================================================================================================================================================
代码说明
============================================================================================================================================================
公式：softmax(x_i) = exp(x_i - x_max) / sum_j{ exp(x_j - x_max) }
算法：
1. 计算全局最大值 m（逐块用 vsetvl 读取向量并做向量归约 vfredmax）。
2. 对每个元素 x_i 减去 m，然后计算 exp(x_i - m)。注意：RVV 没有标准硬件向量 exp，通常两种策略：
    实现向量化近似 exp (需要多项式/查表，复杂但速度更快)。
3. 对 exp 值做向量归约求和（vfredosum）。
4. 将每个 exp 除以总和得到最终概率（向量除法或逐元素标量除法）。
LMUL=1
    使用 e32m1（LMUL=1），在 VLEN=512 时每个向量寄存器包含 16 个 e32 元素。
    - 所有向量计算使用 f32m1。
    - 归约仍由向量归约指令完成（f32m1 -> f32m1），最终得到的“标量”结果通过写内存/整数重解释再以向量方式广播回去（避免使用标量浮点寄存器）。示例流程：
        1) 使用 __riscv_vse32_v_u32m1(..., vl=1) 将归约结果写入内存（按位表示）。
        2) 使用 __riscv_vle32_v_u32m1(..., vl=1) 读回整数向量并用 __riscv_vreinterpret_v_u32m1_f32m1 构造 f32m1 向量。
        3) 用 __riscv_vrgather_vi_f32m1(...) 或 vreinterpret+vmv 将该单元素向量广播到当前块长度 vl。
    - 三处主循环（求 max、计算 exp 并累加、归一化）中的 vsetvl 都改为 __riscv_vsetvl_e32m1(avl) 或 __riscv_vsetvl_e32m1(1)（按需）。
    - vfredmax_vs 与 vfredosum_vs 的第三个参数以及传入的 vl 必须使用当前块的 vl（不能直接用 vlmax 作第三参数），以保证归约与块长度一致。
    - 预构建常量向量可用 vlmax = __riscv_vsetvlmax_e32m1() 生成一次，然后用 __riscv_vmv_v_x_u32m1(const_bits, vlmax) 创建常量向量；后续运算仍按当前块 vl 执行。
    - LMUL=1 每次处理 16 个元素。
============================================================================================================================================================
*/ 
#define VLEN 512
#include <riscv_vector.h>
#define INFINITY (__builtin_inff())
#include <stdio.h>
#include <string.h>
#include "data_softmax.h"
#include "uart_helper.c"

#define CLOCK_FREQUENCY 25000000 //50MHz
#define UART_BITRATE    115200  

// 日志缓冲：先存到内存，再延迟打印
#ifndef LOG_MAX
#define LOG_MAX N
#endif

__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_max_x;
__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_sum_raw;
__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_sum_eps;


__attribute__((section(".logbuf"), aligned(64))) static volatile int32_t g_log_pos[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_exp0[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_blk_sum[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile int32_t g_log_norm_pos[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float   g_log_norm_dst0[LOG_MAX];
static int g_log_norm_cnt = 0;
static int g_log_cnt = 0;

// 简单的空转延时函数，使用 asm volatile 防止被优化掉
static inline void debug_delay_cycles(unsigned cycles) {
  for (unsigned i = 0; i < cycles; ++i) {
    asm volatile("nop");
  }
}

static inline void enable_vector_and_fpu(void) {
        // Set mstatus.VS=11 and mstatus.FS=11 so vector/FPU instructions won't trap.
        // 0x6600 sets VS[10:9]=3 and FS[14:13]=3.
        register unsigned long x = 0x6600;
        asm volatile("csrs mstatus, %0" :: "r"(x) : "memory");
}


// Compare helper that only uses uart_helper.c output routines.
static void compare_and_print(const float* got, const float* ref, int n, float tol) {
    int mism = 0;
    float max_err = 0.0f;
    int max_idx = -1;

    print_uart("detail:\r\n");
    for (int i = 0; i < n; ++i) {
        float d = my_fabs(got[i] - ref[i]);
        if (d > max_err) { max_err = d; max_idx = i; }

        // 逐项打印（无论是否越界）
        print_idx_prefix("i", i);
        print_uart(" got=");  print_float_fixed3(got[i]);
        print_uart(" ref=");  print_float_fixed3(ref[i]);
        print_uart(" diff="); print_float_fixed3(d);
        if (d > tol) { print_uart(" FAIL"); ++mism; }
        else         { print_uart(" OK"); }
        print_uart("\r\n");
    }

    // 汇总
    print_uart("summary: mismatches=");
    print_dec32(mism);
    print_uart(" max_err@");
    print_dec32(max_idx);
    print_uart("=");
    print_float_fixed3(max_err);
    print_uart("\r\n");
}
// ========== softmax 相关 ==========
#if 1
static inline uint32_t load_f32_bits(const float* p) {
    uint32_t u; memcpy(&u, p, 4); return u;
}

void softmax_stable_rvv_fp32(float* dst, const float* src, size_t n);
float quick_dirty_vector_expf(float* dst, float* src, float max_x, size_t n);
uint32_t quick_dirty_vector_expf_no_scalar(float* dst, float* src, uint32_t max_x_bits, size_t n);

__attribute__((aligned(64))) float dst[N] = {0};
__attribute__((aligned(64))) float diff_mem[N] = {0};

int main(){
   
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    // 计算
    print_uart("Softmax RVV LMUL=1 Test\r\n");
    enable_vector_and_fpu();
    print_uart("Initializing.....\r\n");
    softmax_stable_rvv_fp32(dst, src, N);
    debug_delay_cycles(1000000);
    // 对比 golden（容差 1e-3）
    print_uart("Compare vs golden:\r\n");
    compare_and_print(dst, golden, N, 1e-3f);

    return 0;
}
// 用内存作为 vrgather.vi 的输出
static inline vfloat32m8_t asm_vrgather_vi_f32m8_mem(vfloat32m8_t src, int imm, size_t vl) {
    float buf[16] __attribute__((aligned(64))); // m8 最多16元素（VLEN=512）
    __riscv_vse32_v_f32m8(buf, src, vl);
    __asm__ volatile (
        "vle32.v v24, (%0)\n"
        "vrgather.vi v24, v25, %1\n"
        "vse32.v v24, (%0)\n"
        :
        : "r"(buf), "I"(imm)
        : "v24", "v25", "memory"
    );
    return __riscv_vle32_v_f32m8(buf, vl);
}

// 用内存作为 vrgather.vi 的输出
static inline vfloat32m1_t asm_vrgather_vi_f32m1_mem(vfloat32m1_t src, int imm, size_t vl) {
    float buf[8] __attribute__((aligned(64))); // m8 最多16元素（VLEN=512）
    __riscv_vse32_v_f32m1(buf, src, vl);
    __asm__ volatile (
        "vle32.v v24, (%0)\n"
        "vrgather.vi v24, v25, %1\n"
        "vse32.v v24, (%0)\n"
        :
        : "r"(buf), "I"(imm)
        : "v24", "v25", "memory"
    );
    return __riscv_vle32_v_f32m1(buf, vl);
}

// static inline void debug_dump_vec_f32(const char* name, size_t pos, vfloat32m1_t v, size_t vl) {
//     float buf[VLEN/32] __attribute__((aligned(64)));  // m1 在 VLEN=512 时最大 16
//     __riscv_vse32_v_f32m1(buf, v, vl);                // 先 vse 存回内存
//     debug_delay_cycles(100000);                       // 延时，确保外设/可视化
//     for (size_t lane = 0; lane < vl; ++lane) {
//         print_idx_prefix(name, (int)(pos + lane));
//         print_float_fixed3(buf[lane]);
//         print_uart("\r\n");
//     }
// }
// 完全避免标量浮点寄存器的版本
void softmax_stable_rvv_fp32(float* dst, const float *src, size_t n)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1(); 
    
    // // 使用内存数组存储所有常量，通过整数指针访问
    static const uint32_t constants[] __attribute__((aligned(64))) = {
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
    
    // //SPIKE打印:输入数据
    // dbg_print_line("Input data:\n");
    // for (size_t i = 0; i < n; i++) {
    //     dbg_print_idx_hex32("src", (uint32_t)i, "bits", load_f32_bits(&src[i]));
    // }

    // ===============Pass-1: 求全局最大值 max_x====================================
    const float* src_orig = src; // 保存原始指针
    size_t avl = n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        // print_uart("[softmax] avl_max_x:");
        // print_dec32((size_t)vl);
        // print_uart("\r\n");
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

    // // 打印 max_x
    // size_t vl = __riscv_vsetvl_e32m1(avl);
    // __riscv_vse32_v_f32m1((float*)&g_log_max_x, vredmax, vl);  // vse 先写回内存
    // debug_delay_cycles(1000000);                                // 延时
    // print_uart("[softmax] ");
    // print_f32_scalar("max_x", (float)g_log_max_x);    

    
    // 通过内存获取 max_x，避免标量浮点寄存器
    uint32_t max_x_bits[1] __attribute__((aligned(64)));
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
        // print_uart("[softmax] avl2:");
        // print_dec32((size_t)vl);
        // print_uart("\r\n");
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

        // UART打印：打印本块每个元素的 p 和 2^k（先写内存→延时→再打印）
        // debug_delay_cycles(1000000);
        // {
        //     float p_buf[VLEN/32]  __attribute__((aligned(64)));
        //     float f2k_buf[VLEN/32] __attribute__((aligned(64)));
        //     __riscv_vse32_v_f32m1(p_buf,  p,    vl);
        //     // __riscv_vse32_v_f32m1(f2k_buf, vf2k, vl);
        //     debug_delay_cycles(100000); // 延时
        //     for (size_t lane = 0; lane < vl; ++lane) {
        //         size_t i = pos + lane;
        //         print_uart("[softmax] p[");
        //         print_dec32((int32_t)(pos + lane));
        //         print_uart("]=");
        //         print_float_fixed3(p_buf[lane]);    // 直接从内存读取刚写入的 exp
        //         print_uart("\r\n");
        //         // print_uart("[softmax] 2^k[");
        //         // print_dec32((int32_t)(pos + lane));
        //         // print_uart("]=");
        //         // print_float_fixed3(f2k_buf[lane]);    // 直接从内存读取刚写入的 exp
        //         // print_uart("\r\n");
        //     }
        // }

        //exp = p * 2^k
        vfloat32m1_t vexp  = __riscv_vfmul_vv_f32m1(p, vf2k, vl);
        // 写出 exp，并累加块和 -> 总和保存在单元素向量 vsum1
        __riscv_vse32_v_f32m1(dst, vexp, vl);
        vfloat32m1_t vblk  = __riscv_vfredosum_vs_f32m1_f32m1(vexp, vzero, vl);
        vsum1 = __riscv_vfadd_vv_f32m1(vsum1, vblk, __riscv_vsetvl_e32m1(1));



        // // UART打印： 打印本块全部 exp（先存到 dst -> 延时 -> 串口打印）
        // debug_delay_cycles(1000000);
        // for (size_t lane = 0; lane < vl; ++lane) {
        //     print_uart("[softmax] exp[");
        //     print_dec32((int32_t)(pos + lane));
        //     print_uart("]=");
        //     print_float_fixed3(dst[lane]);    // 直接从内存读取刚写入的 exp
        //     print_uart("\r\n");
        // }
        //     // UART 打印：块起点、vexp[0]、blk_sum
        //     vfloat32m1_t vfirst = __riscv_vrgather_vx_f32m1(vexp, 0, vl);
        //     float exp0  = v1_to_f32(vfirst);
        //     float blks  = v1_to_f32(vblk);
        //    int idx = g_log_cnt;
        //     if (idx < LOG_MAX) {                   // 先写内存缓存
        //         g_log_pos[idx]     = (int32_t)pos;
        //        __riscv_vse32_v_f32m1((float*)&g_log_exp0[idx],    vfirst, vl); // vse 写回
        //        __riscv_vse32_v_f32m1((float*)&g_log_blk_sum[idx], vblk,   vl); // vse 写回
        //         g_log_cnt = idx + 1;
        //     }
        // debug_delay_cycles(1000000);            // 延时
        // // 从缓存读取并打印
        // if (idx < LOG_MAX) {
        //     print_uart("[softmax] block@"); print_dec32(g_log_pos[idx]);
        //     print_uart(" vexp0=");  print_float_fixed3((float)g_log_exp0[idx]);
        //     print_uart(" blk_sum=");print_float_fixed3((float)g_log_blk_sum[idx]);
        //     print_uart("\r\n");
        // }

        avl2 -= vl;
        src  += vl;
        dst  += vl;
    }

    // // 打印 exp 总和（加 eps 之前/之后）
    // __riscv_vse32_v_f32m1((float*)&g_log_sum_raw, vsum1, vl);  // vse 先写回内存
    // debug_delay_cycles(1000000);              // 延时
    // print_uart("[softmax] "); 
    // print_f32_scalar("sum_raw", (float)g_log_sum_raw);


    // 计算 inv(sum)（单元素向量），避免标量浮点
    size_t vl1 = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vsum1_eps = __riscv_vfadd_vv_f32m1(vsum1, veps1, vl1);
    vfloat32m1_t vinv1 = __riscv_vfrec7_v_f32m1(vsum1_eps, vl1);

    // //print
    // float sum_eps = v1_to_f32(vsum1_eps);
    //__riscv_vse32_v_f32m1((float*)&g_log_sum_eps, vsum1_eps,vl1);                 // 先存
    // debug_delay_cycles(1000000);              // 延时
    // print_uart("[softmax] "); print_f32_scalar("sum_eps", (float)g_log_sum_eps);

    // NR refine ×2：vinv = vinv * (2 - d*vinv)
    vfloat32m1_t corr = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1); // 2 - d*x
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    corr  = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1);
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    // 1. 先把单元素倒数 vinv1 写到 inv_table
    uint32_t inv_table[1] __attribute__((aligned(64)));
    __riscv_vse32_v_u32m1(inv_table, __riscv_vreinterpret_v_f32m1_u32m1(vinv1), 1);
    // 2. 构造单元素向量
    vfloat32m1_t inv_vec = __riscv_vreinterpret_v_u32m1_f32m1(
    __riscv_vle32_v_u32m1(inv_table, 1));

    // =================Pass-3: 归一化（用 vrgather_vx 将单元素 vinv1 广播到当前 vl）=========================
    dst = dst_orig;
    size_t avl3 = n;
    while (avl3 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl3);
        // print_uart("[softmax] avl3_norm:");
        // print_dec32((size_t)vl);
        // print_uart("\r\n");
        size_t pos = n - avl3;
        vfloat32m1_t row   = __riscv_vle32_v_f32m1(dst, vl);
        // 用 vrgather.vi 广播 inv_vec[0] 到所有 lane
        //vfloat32m1_t vinvB = asm_vrgather_vi_f32m1_mem(inv_vec, 0, vl);
        vfloat32m1_t vinvB = __riscv_vrgather_vx_f32m1(inv_vec, 0, vl); //这里编译器会变为vrgather.vi
        // 归一化
        row = __riscv_vfmul_vv_f32m1(row, vinvB, vl);
        __riscv_vse32_v_f32m1(dst, row, vl);

        // // UART 打印：归一化后首元素
        // vfloat32m1_t rfirst = __riscv_vrgather_vx_f32m1(row, 0, vl);
       
        // // 先写入日志缓冲
        // int nidx = g_log_norm_cnt;
        // if (nidx < LOG_MAX) {
        //     g_log_norm_pos[nidx]  = (int32_t)pos;
        //     __riscv_vse32_v_f32m1((float*)&g_log_norm_dst0[nidx], rfirst, vl);
        //     g_log_norm_cnt = nidx + 1;
        // }
        // // 延时
        // debug_delay_cycles(1000000);
        // // 再从内存读出并打印到串口
        // if (nidx < LOG_MAX) {
        //     print_uart("[softmax] norm@"); 
        //     print_dec32(g_log_norm_pos[nidx]);
        //     print_uart(" dst0="); 
        //     print_float_fixed3((float)g_log_norm_dst0[nidx]);
        //     print_uart("\r\n");
        // }

        avl3 -= vl;
        dst  += vl;
    }
    //UART打印：最终结果
    dst = dst_orig;
    //debug_delay_cycles(1000);
    size_t avl4 = n;
    while (avl4 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl4);
        vfloat32m1_t row2 = __riscv_vle32_v_f32m1(dst, vl);
        __riscv_vse32_v_f32m1(dst, row2, vl);
        debug_delay_cycles(1000);
        avl4 -= vl;
        dst  += vl;
    }
    dst = dst_orig;


    // // SPIKE打印：最后结果
    // dbg_print_line("Final results:\n");
    // for (size_t i = 0; i < n; i++) {
    //     dbg_print_idx_hex32("dst", (uint32_t)i, "bits", load_f32_bits(&dst_orig[i]));
    //     dbg_print_idx_hex32("golden", (uint32_t)i, "bits", load_f32_bits(&golden[i]));
    // }
//    dbg_print_hex32("Returning sum bits", sum_bits);

   
}

#endif // softmax related

