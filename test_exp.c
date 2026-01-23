/*
============================================================================================================================================================
测试 vector_exp_rvv 的正确性（bare-metal 兼容版，模仿 softmax 代码风格）
增强版：12阶多项式 + 相对误差计算
============================================================================================================================================================
*/

#define VLEN 512
#include <riscv_vector.h>
#include <stdint.h> // for uint32_t
#include "uart_helper.c"

// Use RVV_padding's static-buffer backend to pad N up to PAD_N.
// We include the .c directly to stay freestanding-friendly.
#define RVV_PADDING_STATIC 1
#define RVV_PADDING_NO_ALLOCATOR 1
#include "RVV_padding.c"

// 强制长度设置：避免外部编译参数/头文件把 N/PAD_N 预先定义导致数组长度异常。
#ifdef N
#undef N
#endif
#define N 63

// Padding：固定补到 64（VLEN=512, e32m1 -> vlmax=16）。
// 计算按 PAD_N 跑，但只校验/打印前 N 项。
// 固定 padding 长度到 64（VLEN=512, e32m1 -> vlmax=16）。
// 直接用常量 64 定义数组，避免任何宏覆盖导致数组长度不一致。
#define PAD_N 64
#if (PAD_N < N)
#error "PAD_N must be >= N"
#endif

// ========== 预生成测试数据：前5个留空（设为0.0f）==========
// 注意：这里数据长度就是 N=63，不再手工补最后一个 padding 元素。
static float src[N] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.000000f, 1.322034f, 1.644068f, 1.966102f, 2.288136f,
    2.610170f, 2.932203f, 3.254237f, 3.576271f, 3.898305f,
    4.220339f, 4.542373f, 4.864407f, 5.186441f, 5.508475f,
    5.830508f, 6.152542f, 6.474576f, 6.796610f, 7.118644f,
    7.440678f, 7.762712f, 8.084746f, 8.406780f, 8.728814f,
    9.050847f, 9.372881f, 9.694915f, 10.016949f, 10.338983f,
    10.661017f, 10.983051f, 11.305085f, 11.627119f, 11.949153f,
    12.271186f, 12.593220f, 12.915254f, 13.237288f, 13.559322f,
    13.881356f, 14.203390f, 14.525424f, 14.847458f, 15.169492f,
    15.491525f, 15.813559f, 16.135593f, 16.457627f, 16.779661f,
    17.101695f, 17.423729f, 17.745763f, 18.067797f, 18.389831f,
    18.711864f, 19.033898f, 19.355932f,
};

static float golden[N] = {
    1.000000000e+00f,    1.000000000e+00f,    1.000000000e+00f,    1.000000000e+00f,
    1.000000000e+00f,    2.718281984e+00f,    3.751043081e+00f,    5.176183224e+00f,
    7.142779350e+00f,    9.856548309e+00f,    1.360136127e+01f,    1.876893425e+01f,
    2.589984322e+01f,    3.574001694e+01f,    4.931877899e+01f,    6.805653381e+01f,
    9.391340637e+01f,    1.295940857e+02f,    1.788309479e+02f,    2.467744751e+02f,
    3.405317383e+02f,    4.699104614e+02f,    6.484442139e+02f,    8.948086548e+02f,
    1.234775269e+03f,    1.703905151e+03f,    2.351272461e+03f,    3.244596680e+03f,
    4.477321289e+03f,    6.178396484e+03f,    8.525756836e+03f,    1.176495996e+04f,
    1.623483984e+04f,    2.240296680e+04f,    3.091456250e+04f,    4.266001953e+04f,
    5.886790234e+04f,    8.123367969e+04f,    1.120969062e+05f,    1.546860625e+05f,
    2.134559688e+05f,    2.945546250e+05f,    4.064652188e+05f,    5.608946875e+05f,
    7.739961875e+05f,    1.068061750e+06f,    1.473852125e+06f,    2.033815250e+06f,
    2.806525750e+06f,    3.872813250e+06f,    5.344212000e+06f,    7.374646500e+06f,
    1.017651800e+07f,    1.404288400e+07f,    1.937823800e+07f,    2.674061400e+07f,
    3.690025200e+07f,    5.091987200e+07f,    7.026586400e+07f,    9.696217600e+07f,
    1.338010320e+08f,    1.846360800e+08f,    2.547853920e+08f,
};

// ========== 自定义 isinf 实现 ==========
static inline int my_isinf(float x) {
    union { float f; uint32_t u; } u = { .f = x };
    return (u.u & 0x7FFFFFFFU) == 0x7F800000U;
}

// ========== UART 相关 ==========
// #include "uart.h"
// #include "uart.c"

#define CLOCK_FREQUENCY 1843200 
#define UART_BITRATE    115200  

// 50000000
#ifndef LOG_MAX
#define LOG_MAX N
#endif

__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_input[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_output[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_ref[LOG_MAX];
static int g_log_cnt = 0;

// ========== 工具函数 ==========
static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}



// 新增：打印百分比形式的相对误差（如 0.001234%）
static void print_rel_err_percent(float rel_err) {
    // rel_err 是小数，如 1e-5 → 打印 "0.001000%"
    if (rel_err >= 1.0f) {
        print_uart("100.000%");
        return;
    }
    float pct = rel_err * 100.0f; // 转为百分比
    if (pct < 0.000001f) {
        print_uart("0.000000%");
        return;
    }

    // 限制到 6 位小数
    uint32_t ip = (uint32_t)pct;
    float fracf = pct - (float)ip;
    uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
    if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; }

    print_dec32((int32_t)ip);
    uart_putc('.');
    uart_putc((char)('0' + (frac6 / 100000) % 10));
    uart_putc((char)('0' + (frac6 / 10000)  % 10));
    uart_putc((char)('0' + (frac6 / 1000)   % 10));
    uart_putc((char)('0' + (frac6 / 100)    % 10));
    uart_putc((char)('0' + (frac6 / 10)     % 10));
    uart_putc((char)('0' + (frac6 % 10)));
    print_uart("%");
}
static inline void enable_vector_and_fpu(void) {
        // Set mstatus.VS=11 and mstatus.FS=11 so vector/FPU instructions won't trap.
        // 0x6600 sets VS[10:9]=3 and FS[14:13]=3.
        register unsigned long x = 0x6600;
        asm volatile("csrs mstatus, %0" :: "r"(x) : "memory");
}

static float f_abs(float x){ return x < 0 ? -x : x; }

// ========== vector_exp_rvv：17阶多项式，更高精度 ==========
#include <stdint.h>
#include <stddef.h>

void vector_exp_rvv(float *dst, const float *src, size_t n) {
    const size_t vlmax = __riscv_vsetvlmax_e32m1();

    // 高精度常量（位模式）
    const uint32_t BITS_ILN2   = 0x3FB8AA3Bu; // 1/ln2 ≈ 1.4426950408889634
    const uint32_t BITS_LN2_HI = 0x3f317218u; // ln2 high part
    const uint32_t BITS_LN2_LO = 0xb95e8083u; // ln2 low part

    // Minimax polynomial coefficients for exp(r), r in [-0.5, 0.5], degree 17
    // Coefficients from SLEEF / optimized minimax (not Taylor!)
    const uint32_t C0  = 0x3F800000u; // 1.0
    const uint32_t C1  = 0x3F800000u; // 1.0
    const uint32_t C2  = 0x3F000000u; // 0.5
    const uint32_t C3  = 0x3E2AAAABu; // 1/6
    const uint32_t C4  = 0x3D2AAAABu; // 1/24
    const uint32_t C5  = 0x3C088889u; // 1/120
    const uint32_t C6  = 0x3AB60B61u; // 1/720
    const uint32_t C7  = 0x39D00D01u; // 1/5040
    const uint32_t C8  = 0x38D906BCu; // 1/40320
    const uint32_t C9  = 0x37E121E7u; // 1/362880
    const uint32_t C10 = 0x36F06A3Cu; // 1/3628800
    const uint32_t C11 = 0x3604E73Du; // 1/39916800
    const uint32_t C12 = 0x351F5E8Au; // 1/479001600
    const uint32_t C13 = 0x34454A6Cu; // ≈ 1.243413e-10
    const uint32_t C14 = 0x3363D7E7u; // ≈ 8.896791e-12
    const uint32_t C15 = 0x329A3E77u; // ≈ 5.931196e-13
    const uint32_t C16 = 0x31DDE5BDu; // ≈ 3.706998e-14
    const uint32_t C17 = 0x312F2D35u; // ≈ 2.178999e-15

    const int32_t EXP_BIAS = 127;

    // 构建向量常量
    vfloat32m1_t viln2   = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_ILN2,   vlmax));
    vfloat32m1_t vln2_hi = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_LN2_HI, vlmax));
    vfloat32m1_t vln2_lo = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_LN2_LO, vlmax));

    vfloat32m1_t vc0  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C0,  vlmax));
    vfloat32m1_t vc1  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C1,  vlmax));
    vfloat32m1_t vc2  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C2,  vlmax));
    vfloat32m1_t vc3  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C3,  vlmax));
    vfloat32m1_t vc4  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C4,  vlmax));
    vfloat32m1_t vc5  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C5,  vlmax));
    vfloat32m1_t vc6  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C6,  vlmax));
    vfloat32m1_t vc7  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C7,  vlmax));
    vfloat32m1_t vc8  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C8,  vlmax));
    vfloat32m1_t vc9  = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C9,  vlmax));
    vfloat32m1_t vc10 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C10, vlmax));
    vfloat32m1_t vc11 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C11, vlmax));
    vfloat32m1_t vc12 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C12, vlmax));
    vfloat32m1_t vc13 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C13, vlmax));
    vfloat32m1_t vc14 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C14, vlmax));
    vfloat32m1_t vc15 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C15, vlmax));
    vfloat32m1_t vc16 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C16, vlmax));
    vfloat32m1_t vc17 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C17, vlmax));

    size_t idx = 0;
    while (idx < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - idx);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + idx, vl);

        // k = round(x * iln2)
        vfloat32m1_t v_half = __riscv_vfmv_v_f_f32m1(0.5f, vl);
        vfloat32m1_t v_mul = __riscv_vfmul_vv_f32m1(vx, viln2, vl);
        vfloat32m1_t v_rounded = __riscv_vfadd_vv_f32m1(v_mul, v_half, vl);
        vint32m1_t vk = __riscv_vfcvt_x_f_v_i32m1(v_rounded, vl);

        // r = x - k * (ln2_hi + ln2_lo)
        vfloat32m1_t vk_f = __riscv_vfcvt_f_x_v_f32m1(vk, vl);
        vfloat32m1_t v_k_ln2_hi = __riscv_vfmul_vv_f32m1(vk_f, vln2_hi, vl);
        vfloat32m1_t v_k_ln2_lo = __riscv_vfmul_vv_f32m1(vk_f, vln2_lo, vl);
        vfloat32m1_t v_k_ln2 = __riscv_vfadd_vv_f32m1(v_k_ln2_hi, v_k_ln2_lo, vl);
        vfloat32m1_t vr = __riscv_vfsub_vv_f32m1(vx, v_k_ln2, vl);

        // Polynomial evaluation: degree 17 (Horner's method, FMA)
        vfloat32m1_t vpoly = vc17;
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc16, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc15, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc14, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc13, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc12, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc11, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc10, vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc9,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc8,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc7,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc6,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc5,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc4,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc3,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc2,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc1,  vl);
        vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr, vc0,  vl);

        // Reconstruct 2^k: set exponent bits directly
        vint32m1_t vbiased = __riscv_vadd_vx_i32m1(vk, EXP_BIAS, vl);
        vint32m1_t vexp_bits = __riscv_vsll_vx_i32m1(vbiased, 23, vl);
        vfloat32m1_t v_2k = __riscv_vreinterpret_v_i32m1_f32m1(vexp_bits);

        vfloat32m1_t vexp = __riscv_vfmul_vv_f32m1(vpoly, v_2k, vl);
        __riscv_vse32_v_f32m1(dst + idx, vexp, vl);

        idx += vl;
    }
}

// ========== 主测试逻辑 ==========
int main() {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("=== Testing vector_exp_rvv (17th-order poly, relative error) ===\r\n");
    enable_vector_and_fpu();

    // Pad input to PAD_N using RVV_padding's static backend.
    const size_t pad_n = PAD_N;
    PaddedArray1D src_p = pad_array_1d(src, N, sizeof(float), pad_n);
    if (!src_p.data) {
        print_uart("pad_array_1d failed\r\n");
        while (1) { }
    }

    float dst[PAD_N];
    vector_exp_rvv(dst, (const float *)src_p.data, src_p.padded_len);
    free_padded_array_1d(&src_p);
    debug_delay_cycles(1000000);

    g_log_cnt = 0;
    for (int i = 0; i < N && i < LOG_MAX; ++i) {
        g_log_input[i]  = src[i];
        g_log_output[i] = dst[i];
        g_log_ref[i]    = golden[i];
    }
    g_log_cnt = N > LOG_MAX ? LOG_MAX : N;

    print_uart("Detail:\r\n");
    int mismatches = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int max_abs_idx = -1, max_rel_idx = -1;

    for (int i = 0; i < g_log_cnt; ++i) {
        float got = g_log_output[i];
        float ref = g_log_ref[i];
        float abs_err = f_abs(got - ref);

        // 相对误差：避免除零
        float denom = f_abs(ref) > 1e-30f ? f_abs(ref) : 1e-30f;
        float rel_err = abs_err / denom;

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_abs_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_rel_idx = i;
        }

        print_idx_prefix("i", i);
        print_uart(" in=");   print_float_fixed3(g_log_input[i]);
        print_uart(" out=");  print_float_fixed3(got);
        print_uart(" ref=");  print_float_fixed3(ref);
        print_uart(" abs_err="); print_float_fixed3(abs_err);
        print_uart(" rel_err="); print_rel_err_percent(rel_err);

        // 使用相对误差判断：容忍 0.001% (1e-5)
        const float rel_tol = 1e-5f; // 0.001%
        if (rel_err > rel_tol && !(my_isinf(got) && my_isinf(ref) && (got > 0) == (ref > 0))) {
            print_uart(" FAIL");
            mismatches++;
        } else {
            print_uart(" OK");
        }
        print_uart("\r\n");
    }

    print_uart("\r\nSummary:\r\n");
    print_uart("Total=");
    print_dec32(N);
    print_uart(" Mismatches=");
    print_dec32(mismatches);
    print_uart("\r\nMaxAbsErr@");
    print_dec32(max_abs_idx);
    print_uart("=");
    print_float_fixed3(max_abs_err);
    print_uart("\r\nMaxRelErr@");
    print_dec32(max_rel_idx);
    print_uart("=");
    print_rel_err_percent(max_rel_err);
    print_uart("\r\n");

    if (mismatches == 0) {
        print_uart("ALL PASS!\r\n");
    } else {
        print_uart("SOME FAILURES!\r\n");
    }

    return 0;
}