#ifndef RISCV_VECTOR_KERNELS_H
#define RISCV_VECTOR_KERNELS_H

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

static inline void softmax(float* x, int size) {
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
    uint32_t max_x_bits[1];
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
    uint32_t inv_table[1];
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
}

#endif
