/*
============================================================================================================================================================
测试 vector_cos_rvv 的正确性（bare-metal 兼容版 + 标量参考对比 + x_base 调试 + vl=16）
- 所有向量操作使用 vl = 16
- 单元素计算：只使用向量第0个元素，其余为 don't-care
- 调试打印仅对全局索引 5~9
- 新增打印 x_base
- 同时记录向量结果和标量参考结果
- ✅ 所有调试提取均先 store 到内存 + delay(5000) 确保 FPGA 可见
- ✅ vfmv.v.f + vse32 链路自检使用 vl=16
- ✅ 所有 buffer 至少 16 元素，aligned(64)
============================================================================================================================================================
*/

#define VLEN 512
#include <riscv_vector.h>
#include <stdint.h>

#ifndef N
#define N 32
#endif

// ========== 控制宏 ==========
//#define USE_VL2_FOR_DEBUG   // ← 已弃用（强制 vl=16）
#define DEBUG_PRINT_X_BASE    // ← 始终启用 x_base 打印（可注释关闭）

// ========== 输入数据 ==========
static const float src[N] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,
    0.1f, -0.1f, 0.25f, -0.25f, 0.5f,
    -0.5f, 0.78539816339f, -0.78539816339f, 1.0f, -1.0f,
    1.5f, -1.5f, 2.0f, -2.0f, 3.0f, -3.0f,
    3.1415926535f, -3.1415926535f, 4.7123889804f, -4.7123889804f,
    6.0f, -6.0f, 10.0f, -10.0f, 20.0f, -20.0f
};

// ========== 参考值 ==========
static const float golden[N] = {
    1.000000f,
    1.000000f,
    1.000000f,
    1.000000f,
    1.000000f,
    1.000000f,
    9.950042e-1f,
    9.950042e-1f,
    9.689124e-1f,
    9.689124e-1f,
    8.775826e-1f,
    8.775826e-1f,
    7.071068e-1f,
    7.071068e-1f,
    5.403023e-1f,
    5.403023e-1f,
    7.073720e-02f,
    7.073720e-02f,
    -4.161468e-1f,
    -4.161468e-1f,
    -9.899925e-1f,
    -9.899925e-1f,
    -1.000000f,
    -1.000000f,
    0.0f,
    0.0f,
    9.601703e-1f,
    9.601703e-1f,
    -8.390715e-1f,
    -8.390715e-1f,
    4.080821e-1f,
    4.080821e-1f
};

// ========== 自定义 isinf/isnan ==========
static inline int my_isinf(float x) {
    union { float f; uint32_t u; } u = { .f = x };
    return (u.u & 0x7FFFFFFFU) == 0x7F800000U;
}
static inline int my_isnan(float x) {
    union { float f; uint32_t u; } u = { .f = x };
    return (u.u & 0x7FFFFFFFU) > 0x7F800000U;
}

// ========== UART ==========
#include "uart.h"
#include "uart.c"

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200  

#ifndef LOG_MAX
#define LOG_MAX N
#endif

__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_input[LOG_MAX];
__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_output_vec[LOG_MAX];     // 向量结果
__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_output_scalar[LOG_MAX];  // 标量结果
__attribute__((section(".logbuf"), aligned(64))) static volatile float g_log_ref[LOG_MAX];
static int g_log_cnt = 0;

// ========== 工具函数 ==========
static inline void debug_delay_cycles(unsigned cycles) {
    asm volatile("" ::: "memory"); // 编译器屏障
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}

static inline void uart_putc(char c) { 
    write_serial((uint8_t)c); 
}

static void print_dec32(int32_t v) {
    char buf[16]; int i = 0;
    if (v == 0) { uart_putc('0'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    while (v && i < (int)sizeof(buf)) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i--) uart_putc(buf[i]);
}

static void print_rel_err_percent(float rel_err) {
    if (rel_err >= 1.0f) {
        print_uart("100.000%");
        debug_delay_cycles(5000);
        return;
    }
    float pct = rel_err * 100.0f;
    if (pct < 0.000001f) {
        print_uart("0.000000%");
        debug_delay_cycles(5000);
        return;
    }

    uint32_t ip = (uint32_t)pct;
    float fracf = pct - (float)ip;
    uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
    if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; }

    print_dec32((int32_t)ip);
    debug_delay_cycles(5000);
    uart_putc('.');
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 100000) % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 10000)  % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 1000)   % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 100)    % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 10)     % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 % 10)));
    debug_delay_cycles(5000);
    print_uart("%");
    debug_delay_cycles(5000);
}

static void print_float_fixed3(float x) {
    union { float f; uint32_t u; } xu = { .f = x };
    uint32_t u = xu.u, exp = u & 0x7F800000u, frac = u & 0x007FFFFFu, sign = u >> 31;
    if (exp == 0x7F800000u) {
        if (frac == 0) { print_uart(sign ? "-inf" : "inf"); debug_delay_cycles(5000); return; }
        print_uart("nan"); debug_delay_cycles(5000); return;
    }
    if ((u & 0x7FFFFFFFU) == 0) {
        print_uart("0.000000"); debug_delay_cycles(5000); return;
    }

    int neg = (x < 0.0f);
    if (neg) x = -x;

    uint32_t ip = (uint32_t)x;
    float fracf = x - (float)ip;
    uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
    if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; }
    if (neg) { uart_putc('-'); debug_delay_cycles(5000); }
    print_dec32((int32_t)ip);
    debug_delay_cycles(5000);
    uart_putc('.');
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 100000) % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 10000)  % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 1000)   % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 100)    % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 / 10)     % 10));
    debug_delay_cycles(5000);
    uart_putc((char)('0' + (frac6 % 10)));
    debug_delay_cycles(5000);
}

static void print_idx_prefix(const char* name, int i) {
    print_uart(name); debug_delay_cycles(5000);
    uart_putc('['); debug_delay_cycles(5000);
    print_dec32(i); debug_delay_cycles(5000);
    uart_putc(']'); debug_delay_cycles(5000);
    uart_putc('='); debug_delay_cycles(5000);
}

static float f_abs(float x){ return x < 0 ? -x : x; }

// ========== 数学常量 ==========
static const volatile float COS_INV_TWO_PI = 0.15915494309189535f;
static const float COS_TWO_PI     = 6.2831853071795865f;
static const float COS_PI         = 3.1415926535897932f;
static const float COS_HALF_PI    = 1.5707963267948966f;
static const float C0 = 1.0f;
static const float C2 = -0.5f;
static const float C4 = 0.0416666667f;
static const float C6 = -0.0013888889f;

// ========== 纯标量 cos 近似 ==========
static float scalar_cos_approx(float x) {
    float t = x * COS_INV_TWO_PI;
    float t_rounded = (t >= 0.0f) ? (t + 0.5f) : (t - 0.5f);
    int32_t k = (int32_t)t_rounded;
    float xr = x - (float)k * COS_TWO_PI;
    float ax = (xr < 0.0f) ? -xr : xr;
    int flip = (ax > COS_HALF_PI);
    float x_base = flip ? (COS_PI - ax) : ax;
    float sign = flip ? -1.0f : 1.0f;

    float x2 = x_base * x_base;
    float p = C6 * x2 + C4;
    p = p * x2 + C2;
    p = p * x2 + C0;
    return sign * p;
}

// ========== 高精度 2π 拆分常量 ==========
static const float COS_TWO_PI_HI = 6.2831855f;
static const float COS_TWO_PI_LO = -4.1015625e-7f;

// ========================================================================
// ✅ 最小测试：验证 vfmv.v.f + vse32 链路（vl=16）
void test_vfmv_vse32_chain(void) {
    print_uart("\r\n=== [TEST] vfmv.v.f + vse32 Chain (vl=16) ===\r\n");
    debug_delay_cycles(5000);

    const float input_val = 0.123456f;

    size_t vl = __riscv_vsetvl_e32m1(16);
    if (vl == 0 || vl > 16) {
        print_uart("[FAIL] Invalid vl!\r\n");
        debug_delay_cycles(5000);
        return;
    }

    vfloat32m1_t v = __riscv_vfmv_v_f_f32m1(input_val, vl);

    // Buffer 必须 ≥ vl
    volatile float output_buf[16] __attribute__((aligned(64)));
    volatile float temp_extract[16] __attribute__((aligned(64)));

    __riscv_vse32_v_f32m1((float*)output_buf, v, vl);
    debug_delay_cycles(5000);
    float result_from_store = output_buf[0];

    __riscv_vse32_v_f32m1((float*)temp_extract, v, vl);
    debug_delay_cycles(5000);
    float result_from_extract = temp_extract[0];

    print_uart("Input scalar      = ");
    print_float_fixed3(input_val);
    print_uart("\r\n"); debug_delay_cycles(5000);

    print_uart("Output via vse32  = ");
    print_float_fixed3(result_from_store);
    print_uart("\r\n"); debug_delay_cycles(5000);

    print_uart("Output via extract= ");
    print_float_fixed3(result_from_extract);
    print_uart("\r\n"); debug_delay_cycles(5000);

    union { float f; uint32_t u; } in = { .f = input_val };
    union { float f; uint32_t u; } out1 = { .f = result_from_store };
    union { float f; uint32_t u; } out2 = { .f = result_from_extract };

    if (in.u == out1.u && in.u == out2.u) {
        print_uart("[PASS] vfmv.v.f + vse32 chain works correctly!\r\n");
        debug_delay_cycles(5000);
    } else {
        print_uart("[FAIL] Data corruption detected!\r\n");
        debug_delay_cycles(5000);
    }

    print_uart("=== END TEST ===\r\n");
    debug_delay_cycles(5000);
}

// ========== 修正后的 vector_cos_rvv_f32m1（vl=16） ==========
void vector_cos_rvv_f32m1(vfloat32m1_t* out, const vfloat32m1_t x, size_t vl, int should_debug) {
    vfloat32m1_t t = __riscv_vfmul_vf_f32m1(x, COS_INV_TWO_PI, vl);

    if (should_debug) {
        float t_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(t_val, t, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: t=");
        print_float_fixed3(t_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t t_rounded = __riscv_vfadd_vf_f32m1(t, 0.5f, vl);
    vint32m1_t k = __riscv_vfcvt_x_f_v_i32m1(t_rounded, vl);
    vfloat32m1_t k_f = __riscv_vfcvt_f_x_v_f32m1(k, vl);

    if (should_debug) {
        float kf_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(kf_val, k_f, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: k_f=");
        print_float_fixed3(kf_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t v_2pi_hi = __riscv_vfmv_v_f_f32m1(COS_TWO_PI_HI, vl);
    vfloat32m1_t v_2pi_lo = __riscv_vfmv_v_f_f32m1(COS_TWO_PI_LO, vl);

    vfloat32m1_t v_k_2pi_hi = __riscv_vfmul_vv_f32m1(k_f, v_2pi_hi, vl);
    vfloat32m1_t v_k_2pi_lo = __riscv_vfmul_vv_f32m1(k_f, v_2pi_lo, vl);
    vfloat32m1_t v_k_2pi = __riscv_vfadd_vv_f32m1(v_k_2pi_hi, v_k_2pi_lo, vl);

    if (should_debug) {
        float vk2pi_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(vk2pi_val, v_k_2pi, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: v_k_2pi=");
        print_float_fixed3(vk2pi_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t xr = __riscv_vfsub_vv_f32m1(x, v_k_2pi, vl);

    if (should_debug) {
        float xr_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(xr_val, xr, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: xr=");
        print_float_fixed3(xr_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t ax = __riscv_vfabs_v_f32m1(xr, vl);

    if (should_debug) {
        float ax_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(ax_val, ax, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: ax=");
        print_float_fixed3(ax_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vbool32_t flip = __riscv_vmfgt_vf_f32m1_b32(ax, COS_HALF_PI, vl);

    vfloat32m1_t one_f  = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t zero_f = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t flip_f = __riscv_vmerge_vvm_f32m1(one_f, zero_f, flip, vl);

    if (should_debug) {
        float flip_val_arr[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(flip_val_arr, flip_f, vl);
        debug_delay_cycles(5000);
        float flip_val = flip_val_arr[0];
        print_uart("DEBUG: flip=");
        print_uart(flip_val > 0.5f ? "true" : "false");
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t pi_vec = __riscv_vfmv_v_f_f32m1(COS_PI, vl);
    vfloat32m1_t pi_minus_ax = __riscv_vfsub_vv_f32m1(pi_vec, ax, vl);

    vfloat32m1_t one_minus_flip = __riscv_vfsub_vv_f32m1(one_f, flip_f, vl);
    vfloat32m1_t term1 = __riscv_vfmul_vv_f32m1(flip_f, pi_minus_ax, vl);
    vfloat32m1_t term2 = __riscv_vfmul_vv_f32m1(one_minus_flip, ax, vl);
    vfloat32m1_t x_base = __riscv_vfadd_vv_f32m1(term1, term2, vl);

    if (should_debug) {
        float xb_val[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(xb_val, x_base, vl);
        debug_delay_cycles(5000);
        print_uart("DEBUG: x_base=");
        print_float_fixed3(xb_val[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, vl);
    vfloat32m1_t term_sign1 = __riscv_vfmul_vv_f32m1(flip_f, neg_one, vl);
    vfloat32m1_t term_sign2 = __riscv_vfmul_vv_f32m1(one_minus_flip, one_f, vl);
    vfloat32m1_t sign = __riscv_vfadd_vv_f32m1(term_sign1, term_sign2, vl);

    vfloat32m1_t x2 = __riscv_vfmul_vv_f32m1(x_base, x_base, vl);

    vfloat32m1_t p = __riscv_vfmul_vf_f32m1(x2, C6, vl);
    p = __riscv_vfadd_vf_f32m1(p, C4, vl);
    p = __riscv_vfmul_vv_f32m1(p, x2, vl);
    p = __riscv_vfadd_vf_f32m1(p, C2, vl);
    p = __riscv_vfmul_vv_f32m1(p, x2, vl);
    p = __riscv_vfadd_vf_f32m1(p, C0, vl);

    *out = __riscv_vfmul_vv_f32m1(sign, p, vl);
}

// ========== 向量化的单元素 cos 实现（使用 vl=16 上下文） ==========
float vector_cos_rvv(float x, int idx) {
    int should_debug = (idx >= 5 && idx <= 9);
    size_t vl = __riscv_vsetvl_e32m1(16);  // 强制 vl=16

    // 构造向量：仅第0元素为 x，其余为 0（不影响结果）
    vfloat32m1_t vx = __riscv_vfmv_v_f_f32m1(x, vl);

    if (should_debug) {
        print_uart("\r\n=== DEBUG START (idx=");
        print_dec32(idx);
        print_uart(", vl=");
        print_dec32((int)vl);
        print_uart(") ===\r\n");
        debug_delay_cycles(5000);

        print_uart("Input x=");
        print_float_fixed3(x);
        print_uart("\r\n"); debug_delay_cycles(5000);

        float vx_arr[16] __attribute__((aligned(64)));
        __riscv_vse32_v_f32m1(vx_arr, vx, vl);
        debug_delay_cycles(5000);
        print_uart("Loaded vx[0] = ");
        print_float_fixed3(vx_arr[0]);
        print_uart("\r\n"); debug_delay_cycles(5000);
    }

    vfloat32m1_t result_vec;
    vector_cos_rvv_f32m1(&result_vec, vx, vl, should_debug);

    float res_arr[16] __attribute__((aligned(64)));
    __riscv_vse32_v_f32m1(res_arr, result_vec, vl);
    debug_delay_cycles(5000);
    float vec_result = res_arr[0];
    float scalar_ref = scalar_cos_approx(x);

    if (should_debug) {
        print_uart("Vector result = ");
        print_float_fixed3(vec_result);
        print_uart("\r\n"); debug_delay_cycles(5000);
        print_uart("Scalar ref    = ");
        print_float_fixed3(scalar_ref);
        print_uart("\r\n"); debug_delay_cycles(5000);
        print_uart("Diff          = ");
        print_float_fixed3(f_abs(vec_result - scalar_ref));
        print_uart("\r\n=== DEBUG END ===\r\n");
        debug_delay_cycles(5000);
    }

    return vec_result;
}

// ========== 主测试 ==========
int main() {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    
    test_vfmv_vse32_chain();
    debug_delay_cycles(100000);

    print_uart("\r\n=== Testing vector_cos_rvv (vl=16, single-element usage) ===\r\n");
    debug_delay_cycles(5000);

    float dst_vec[N];
    float dst_scalar[N];

    for (int i = 0; i < N; ++i) {
        dst_vec[i] = vector_cos_rvv(src[i], i);
        dst_scalar[i] = scalar_cos_approx(src[i]);
    }

    debug_delay_cycles(1000000);

    g_log_cnt = 0;
    for (int i = 0; i < N && i < LOG_MAX; ++i) {
        g_log_input[i]        = src[i];
        g_log_output_vec[i]   = dst_vec[i];
        g_log_output_scalar[i]= dst_scalar[i];
        g_log_ref[i]          = golden[i];
    }
    g_log_cnt = (N > LOG_MAX) ? LOG_MAX : N;

    print_uart("\r\nResult Comparison (using Vector Result):\r\n");
    debug_delay_cycles(5000);

    int mismatches = 0;
    for (int i = 0; i < g_log_cnt; ++i) {
        float got = g_log_output_vec[i];
        float ref = g_log_ref[i];
        float abs_err = f_abs(got - ref);
        float denom = (f_abs(ref) > 1e-30f) ? f_abs(ref) : 1e-30f;
        float rel_err = abs_err / denom;

        print_idx_prefix("i", i);
        print_uart(" vec="); print_float_fixed3(got);
        print_uart(" scalar="); print_float_fixed3(g_log_output_scalar[i]);
        print_uart(" ref="); print_float_fixed3(ref);
        print_uart(" rel_err="); print_rel_err_percent(rel_err);

        if (rel_err > 1e-3f) {
            print_uart(" ❌ FAIL");
            debug_delay_cycles(5000);
            mismatches++;
        } else {
            print_uart(" ✅ OK");
            debug_delay_cycles(5000);
        }
        print_uart("\r\n");
        debug_delay_cycles(5000);
    }

    print_uart("\r\n");
    debug_delay_cycles(5000);
    if (mismatches == 0) {
        print_uart("🎉 ALL PASS!\r\n");
        debug_delay_cycles(5000);
    } else {
        print_uart("💥 ");
        print_dec32(mismatches);
        print_uart(" failures.\r\n");
        debug_delay_cycles(5000);
    }

    return mismatches ? 1 : 0;
}