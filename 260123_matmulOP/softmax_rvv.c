// ---
// author: zhaoxinlei
// version: 1.0
// ---
// [代码说明]
// 本文件实现了基于 RISC-V Vector (RVV) 指令的数值稳定型 Softmax 算法。
// 算法逻辑：
// 1. 通过 RVV 向量化查找输入数组的最大值 (max)。
// 2. 向量化计算 exp(x - max) 并累加得到分母和 (sum)。
// 3. 将每个 exp(x - max) 除以 sum 得到最终概率分布。
// 该实现特别针对大向量长度进行了优化，并保证了在浮点数指数幂计算中的数值稳定性。

#define VLEN 512
#include "riscv_vector.h"
#include <stdint.h>
#include <stddef.h>

static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}

// Helper to load float as bits without type punning issues
// static inline uint32_t load_f32_bits(const float* p) {
//    uint32_t u; memcpy(&u, p, 4); return u;
// }

// Global debug/log buffers removed or commented out for integration

void softmax_stable_rvv_fp32(float* dst, const float *src, size_t n)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1(); 
    
    // Constants
    const uint32_t BITS_NEG_INF = 0xFF800000u;
    const uint32_t BITS_ILN2    = 0x3FB8AA3Bu; // 1/ln(2)
    const uint32_t BITS_LN2_HI  = 0x3F317000u; // ln2 hi
    const uint32_t BITS_LN2_LO  = 0x38800C00u; // ln2 lo
    const uint32_t BITS_EPS     = 0x322BCC77u; // 1e-8
    const uint32_t BITS_TWO     = 0x40000000u; // 2.0

    // e^r polynomial coefficients
    const uint32_t C0 = 0x3F800000u; // 1
    const uint32_t C1 = 0x3F800000u; // 1
    const uint32_t C2 = 0x3F000000u; // 1/2
    const uint32_t C3 = 0x3E2AAAABu; // 1/6
    const uint32_t C4 = 0x3D2AAAABu; // 1/24
    const uint32_t C5 = 0x3C088889u; // 1/120
    const uint32_t C6 = 0x3AB60B61u; // 1/720
    const uint32_t C7 = 0x3A1175D4u; // 1/5040
    const uint32_t C8 = 0x3926ED8Eu; // 1/40320

    // Pre-build constant vectors
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

    const uint32_t val_neg_inf = 0xFF800000u; // Assuming standard IEEE 754 -inf
    vuint32m1_t vneg_inf_int = __riscv_vmv_v_x_u32m1(val_neg_inf, vlmax);
    vfloat32m1_t vmax = __riscv_vreinterpret_v_u32m1_f32m1(vneg_inf_int);
    
    // ===============Pass-1: Find Global Max====================================
    const float* src_orig = src; // Save original pointer
    size_t avl = n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
        vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
        avl -= vl;
        src += vl;
    }
    src = src_orig; // Reset src pointer
    
    // Reduce -> scalar (in element 0)
    vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m1_f32m1(vmax, vnegInf, vlmax);

    // Final maximum reduction
    vfloat32m1_t vredmax = __riscv_vreinterpret_v_u32m1_f32m1(vneg_inf_int);
    vredmax = __riscv_vfredmax_vs_f32m1_f32m1(vmax, vredmax, vlmax);
    
    //=================== Pass-2: exp(x - max_x), write to dst, accumulate sum ========================
    vfloat32m1_t vsum1 = vzero;
    float* dst_orig = dst;
    size_t avl2 = n;
    while (avl2 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl2);
        size_t pos = n - avl2;
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);

        // Broadcast max_x
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

        // p = poly_e(vr) (8th order Horner)
        vfloat32m1_t p = pc8;

        p = __riscv_vfmadd_vv_f32m1(p, vr, pc7, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc6, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc5, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc4, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc3, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc2, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc1, vl);
        p = __riscv_vfmadd_vv_f32m1(p, vr, pc0, vl);
        
        // 2^k Reconstruct: ((k+127)<<23)
        const int exp_bias = 127;
        vint32m1_t vbiased = __riscv_vadd_vx_i32m1(vk, exp_bias, vl);
        vint32m1_t vexp2   = __riscv_vsll_vx_i32m1(vbiased, 23, vl);
        vfloat32m1_t vf2k  = __riscv_vreinterpret_v_i32m1_f32m1(vexp2);

        // exp = p * 2^k
        vfloat32m1_t vexp  = __riscv_vfmul_vv_f32m1(p, vf2k, vl);
        
        // Write exp, and accumulate to block sum
        __riscv_vse32_v_f32m1(dst, vexp, vl);
        vfloat32m1_t vblk  = __riscv_vfredosum_vs_f32m1_f32m1(vexp, vzero, vl);
        vsum1 = __riscv_vfadd_vv_f32m1(vsum1, vblk, __riscv_vsetvl_e32m1(1));

        avl2 -= vl;
        src += vl;
        dst += vl;
    }

    // Compute inv(sum) (single element vector)
    size_t vl1 = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vsum1_eps = __riscv_vfadd_vv_f32m1(vsum1, veps1, vl1);
    vfloat32m1_t vinv1 = __riscv_vfrec7_v_f32m1(vsum1_eps, vl1);

    // NR refine x2
    vfloat32m1_t corr = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1); // 2 - d*x
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    corr  = __riscv_vfnmsac_vv_f32m1(vtwo1, vsum1_eps, vinv1, vl1);
    vinv1 = __riscv_vfmul_vv_f32m1(vinv1, corr, vl1);
    
    // Store back inv to temp buffer for vrgather broadcast? 
    // Actually we can just keep vinv1 in register and use vrgather_vx (scalar broadcast from vector element 0) if supported, 
    // or just use vmul_vs (vector-scalar multiply) if generic intrinsics allow, 
    // but here the code uses vrgather to broadcast element 0 to all lanes.
    
    // In original code:
    // uint32_t inv_table[1];
    // __riscv_vse32_v_u32m1(inv_table, __riscv_vreinterpret_v_f32m1_u32m1(vinv1), 1);
    // vfloat32m1_t inv_vec = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vle32_v_u32m1(inv_table, 1));
    // The previous code did some round trip via memory to ensure correct 'vector' form for broadcast?
    // Let's stick to user's logic but simplified if possible. 
    // Actually, `__riscv_vfmul_vf` is standard for multiply by scalar? 
    // But here `vinv1` is a vector register with the value in element 0.
    
    // =================Pass-3: Normalize=========================
    dst = dst_orig;
    size_t avl3 = n;
    while (avl3 > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl3);
        vfloat32m1_t vdst = __riscv_vle32_v_f32m1(dst, vl);
        
        // Broadcast vinv1[0] to all lanes
        vfloat32m1_t vinv_bcast = __riscv_vrgather_vx_f32m1(vinv1, 0, vl);
        
        vdst = __riscv_vfmul_vv_f32m1(vdst, vinv_bcast, vl);
        __riscv_vse32_v_f32m1(dst, vdst, vl);
        
        avl3 -= vl;
        dst += vl;
    }
    debug_delay_cycles(1000);
}
