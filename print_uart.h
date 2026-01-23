#include "uart.h"
#include "uart.c"
// ========== UART 基础输出 ==========
static inline void uart_putc(char c) { write_serial((uint8_t)c); }

static void print_dec32(int32_t v) {
    char buf[16]; int i = 0;
    if (v == 0) { uart_putc('0'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    while (v && i < (int)sizeof(buf)) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i--) uart_putc(buf[i]);
}


static void print_float_fixed3(float x) {
    // 特殊值：NaN/Inf/-Inf 与 -0.0
    union { float f; uint32_t u; } xu = { .f = x };
    uint32_t u = xu.u, exp = u & 0x7F800000u, frac = u & 0x007FFFFFu, sign = u >> 31;
    if (exp == 0x7F800000u) {               // Inf/NaN
        if (frac == 0) { print_uart(sign ? "-inf" : "inf"); return; }
        print_uart("nan"); return;
    }
    if ((u & 0x7FFFFFFFu) == 0) {           // ±0 -> 0.000000
        print_uart("0.000000"); return;
    }

    int neg = (x < 0.0f);
    if (neg) x = -x;

    // 大数：改用科学计数法，保留 6 位小数
    if (x >= 1000000000.0f) {               // |x| ≥ 1e9 -> S.DDDDDD e±E
        if (neg) uart_putc('-');
        int e = 0;
        // 规范化到 [1,10)
        while (x >= 10.0f) { x *= 0.1f; ++e; }
        while (x < 1.0f)  { x *= 10.0f; --e; }
        uint32_t ip = (uint32_t)x;          // 1..9
        float fracf = x - (float)ip;
        uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
        if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; if (ip >= 10u) { ip = 1u; ++e; } }
        print_dec32((int32_t)ip);
        uart_putc('.');
        // 打印 6 位小数（补零）
        uart_putc((char)('0' + (frac6 / 100000) % 10));
        uart_putc((char)('0' + (frac6 / 10000)  % 10));
        uart_putc((char)('0' + (frac6 / 1000)   % 10));
        uart_putc((char)('0' + (frac6 / 100)    % 10));
        uart_putc((char)('0' + (frac6 / 10)     % 10));
        uart_putc((char)('0' + (frac6 % 10)));
        uart_putc('e');
        if (e >= 0) { uart_putc('+'); print_dec32(e); }
        else        { uart_putc('-'); print_dec32(-e); }
        return;
    }

    // 常规路径：[-]I.FFFFFF（6 位小数，四舍五入）
    uint32_t ip = (uint32_t)x;              // 此处 x < 1e9，安全
    float fracf = x - (float)ip;
    uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
    if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; }
    if (neg) uart_putc('-');
    print_dec32((int32_t)ip);
    uart_putc('.');
    // 打印 6 位小数（补零）
    uart_putc((char)('0' + (frac6 / 100000) % 10));
    uart_putc((char)('0' + (frac6 / 10000)  % 10));
    uart_putc((char)('0' + (frac6 / 1000)   % 10));
    uart_putc((char)('0' + (frac6 / 100)    % 10));
    uart_putc((char)('0' + (frac6 / 10)     % 10));
    uart_putc((char)('0' + (frac6 % 10)));
}



static inline void print_f32_scalar(const char* name, float v) {
    print_uart(name); 
    print_uart("="); 
    print_float_fixed3(v); 
    print_uart("\r\n");
}

// 将 f32m1（单元素向量）转为 float（通过存一元素到内存）
static inline float v1_to_f32(vfloat32m1_t v) {
    uint32_t tmp[1];
    __riscv_vse32_v_u32m1(tmp, __riscv_vreinterpret_v_f32m1_u32m1(v), 1);
    union { uint32_t u; float f; } u = { .u = tmp[0] };
    return u.f;
}


static void print_idx_prefix(const char* name, int i) {
    print_uart(name); 
    uart_putc('['); 
    print_dec32(i); 
    uart_putc(']'); 
    uart_putc('='); 
}

// UART打印数组与比较 
static void print_f32_array(const char* name, const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        print_idx_prefix(name, i);
        print_float_fixed3(a[i]);
        print_uart("\r\n");
    }
}

static float f_abs(float x){ return x < 0 ? -x : x; }

static void compare_and_print(const float* got, const float* ref, int n, float tol) {
    int mism = 0;
    float max_err = 0.0f;
    int max_idx = -1;

    print_uart("detail:\r\n");
    for (int i = 0; i < n; ++i) {
        float d = f_abs(got[i] - ref[i]);
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