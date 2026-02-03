// Bare-metal friendly inference harness for llama2.c
// ---------------------------------------------------
// This variant removes all host OS dependencies so the model can execute in a
// freestanding Spike tohost/fromhost environment. The model and tokenizer
// binaries are embedded directly into the ELF via objcopy; see the Makefile
// rvbare target for how MODEL_BIN/TOKENIZER_BIN become linker symbols used here.

#define MATRIX_KERNEL_DEBUG_PRINT 0
#define MATRIX_KERNEL_DEBUG_PRINT_EVERY 0
#define BARE_USE_CYCLE_COUNTER 1
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "uart_helper.c"
// #include "matrix_kernel.h"
// #include "matrix_kernel_noblk.h"
#include "matrix_kernel_1230.h"
#include "matrix_kernel_noblk_1231.h"
#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

#define BARE_MEM_STATS 1


// Keep -O1 for the overall build, but prevent Clang from auto-vectorizing
// generic code paths (e.g. tokenizer sorting) into RVV instructions.
// This does NOT block explicit RVV/Matrix usage inside the Matrix kernel.
#if defined(__clang__)
#define BARE_NO_AUTOVEC __attribute__((optnone, noinline))
#elif defined(__GNUC__)
#define BARE_NO_AUTOVEC __attribute__((optimize("O0"), noinline))
#else
#define BARE_NO_AUTOVEC
#endif

// Forward declarations for cycle counter helpers.
// These are defined later in the file but used earlier (C99 disallows implicit decls).
#if BARE_USE_CYCLE_COUNTER
static inline uint64_t rdcycle(void);
static inline long cycles_to_ms(uint64_t start, uint64_t end);
#endif


// Tag matmul calls for Matrix-kernel debug prints.
// Enabled only when MATRIX_KERNEL_DEBUG_PRINT=1.
#if MATRIX_KERNEL_DEBUG_PRINT
#define MATMUL_Q_TAGGED(tag, layer, xout, x, w, n, d) \
    do { \
        MATRIX_KERNEL_DEBUG_SET_CONTEXT((tag), (layer)); \
        matmul_q((xout), (x), (w), (n), (d)); \
    } while (0)
#else
#define MATMUL_Q_TAGGED(tag, layer, xout, x, w, n, d) \
    do { \
        (void)(tag); (void)(layer); \
        matmul_q((xout), (x), (w), (n), (d)); \
    } while (0)
#endif
//---------------------------------------------------------------------------
// Quantization helpers (GS = group size)

// Default group size for per-group int8 quantization.
// Can be overridden at build time with e.g. -DGS=64.
#ifndef GS
#define GS 32
#endif

typedef struct {
    int8_t* q;   // quantized values (length n)
    float* s;    // per-group scale factors (length ceil(n/GS))
} QuantizedTensor;

// Forward declarations used by quantization helpers.
// Implementations are provided later in this file.
void exit(int code);
void *calloc(size_t count, size_t size);
static void panic(const char *msg);

static inline float q_fabsf(float x) { return x < 0.0f ? -x : x; }

static inline int8_t q_round_clamp_i8(float x) {
    // Round-to-nearest with halves away from zero (matches roundf semantics).
    int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    return (int8_t)qi;
}

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    const int num_groups = (n + GS - 1) / GS;
    const float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        const int base = group * GS;
        const int count = (base + GS <= n) ? GS : (n - base);

        // find the max absolute value in the current group
        float wmax = 0.0f;
        for (int i = 0; i < count; i++) {
            float val = q_fabsf(x[base + i]);
            if (val > wmax) wmax = val;
        }

        // calculate and write the scaling factor
        float scale = (wmax > 0.0f) ? (wmax / Q_MAX) : 1.0f;
        qx->s[group] = scale;

        // calculate and write the quantized values
        if (wmax == 0.0f) {
            for (int i = 0; i < count; i++) qx->q[base + i] = 0;
        } else {
            float inv = 1.0f / scale;
            for (int i = 0; i < count; i++) {
                qx->q[base + i] = q_round_clamp_i8(x[base + i] * inv);
            }
        }
    }
}

static QuantizedTensor qt_alloc(int n) {
    QuantizedTensor t;
    t.q = (int8_t*)calloc((size_t)n, sizeof(int8_t));
    t.s = (float*)calloc((size_t)((n + GS - 1) / GS), sizeof(float));
    if (!t.q || !t.s) {
        panic("quant buffer alloc failed");
    }
    return t;
}

static void quantize_const(QuantizedTensor *qx, const float* x, int n) {

    const int num_groups = (n + GS - 1) / GS;
    const float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        const int base = group * GS;
        const int count = (base + GS <= n) ? GS : (n - base);

        float wmax = 0.0f;
        for (int i = 0; i < count; i++) {
            float val = q_fabsf(x[base + i]);
            if (val > wmax) wmax = val;
        }

        float scale = (wmax > 0.0f) ? (wmax / Q_MAX) : 1.0f;
        qx->s[group] = scale;

        if (wmax == 0.0f) {
            for (int i = 0; i < count; i++) qx->q[base + i] = 0;
        } else {
            float inv = 1.0f / scale;
            for (int i = 0; i < count; i++) {
                qx->q[base + i] = q_round_clamp_i8(x[base + i] * inv);
            }
        }
    }
}

static void matmul_q(float* xout, const QuantizedTensor *x, const QuantizedTensor *w, int n, int d) {

    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        for (int j = 0; j < n; j += GS) {
            const int count = (j + GS <= n) ? GS : (n - j);
            for (int k = 0; k < count; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
     // W (d,n) @ x (n,) -> xout (d,)
     //
     // This path is fully handled by the Matrix+RVV operator in matrix_kernel.h:
     // - Matrix computes int8xint8 -> int32 dot products (no scalar packing).
     // - RVV clears output and applies per-group scaling/accumulation (ws*xs).
     
     //matrix_kernel_qmatmul_f32_noblk(xout, x->q, x->s, w->q, w->s, n, d, GS, n);
}
// ---------------------------------------------------------------------------
// Tiny libc hooks provided by bare_syscalls.c (no system headers required)
typedef long ssize_t;

void *memcpy(void *dest, const void *src, size_t len);
void *memset(void *dest, int byte, size_t len);
void exit(int code);
int strcmp(const char *s1, const char *s2);
size_t strlen(const char *s);

// ---------------------------------------------------------------------------
// Trap diagnostics
// If a trap happens, crt.S calls handle_trap(cause, epc, regs). If we don't
// provide our own, the default weak handler exits with code -32, which shows as
// "*** FAILED *** (tohost = -32)" in Spike. Provide a strong handler here so
// we can see the real cause.
static inline uintptr_t read_csr_mtval(void) {
    uintptr_t x;
    __asm__ volatile ("csrr %0, mtval" : "=r"(x));
    return x;
}

static inline void enable_rvv_state(void) {
    // Some CRTs start with mstatus.VS=0 (Off). Any RVV instruction (e.g. vsetvli)
    // will then trap with mcause=2 (illegal instruction). Enable VS=Dirty.
    uintptr_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9); // VS[10:9] = 0b11
    mstatus |= (3UL << 13);  // FS=Dirty
    // Some non-standard/accelerator extensions are gated by XS.
    // If XS=Off, their instructions may trap as illegal.
    mstatus |= (3UL << 15);  // XS[16:15] = Dirty
    __asm__ volatile ("csrw mstatus, %0" :: "r"(mstatus) : "memory");
 }

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
    (void)regs;
    uintptr_t mtval = read_csr_mtval();
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("\r\n[trap] mcause=0x");
    print_uart_hex((unsigned long long)cause);
    print_uart(" ("); print_uart_int_dec((unsigned long long)cause); print_uart(")");
    print_uart(" mepc=0x");
    print_uart_hex((unsigned long long)epc);
    print_uart(" mtval=0x");
    print_uart_hex((unsigned long long)mtval);
    print_uart("\r\n");

    // If illegal instruction, provide a hint
    if (((unsigned long long)cause & 0xffULL) == 2ULL) {
        print_uart("[hint] mcause indicates Illegal Instruction (2)\r\n");
    }

    // Dump a few instruction words around mepc to help identify the faulting opcode
    for (int off = -1; off <= 4; ++off) {
        uintptr_t addr = epc + (uintptr_t)(off * 4);
        print_uart("mem["); print_uart_hex((unsigned long long)addr); print_uart("]=0x");
        // Read little-endian 32-bit word; this should be safe for code memory
        uint32_t w = *(const uint32_t *)addr;
        print_uart_hex((unsigned long long)w);
        print_uart("\r\n");
    }

    // Common causes:
    //   2  = illegal instruction
    //   4/6 = load/store address misaligned
    //   5/7 = load/store access fault
    exit(224);
    return epc;
}

// ---------------------------------------------------------------------------
// Build-time knobs (override via e.g. `make rvbare CFLAGS+=-DBARE_STEPS=128`)
#ifndef BARE_PROMPT
#define BARE_PROMPT "Once upon a time" 
#endif
//"Once upon a time"
#ifndef BARE_STEPS
#define BARE_STEPS 16
#endif

#ifndef BARE_TEMPERATURE
#define BARE_TEMPERATURE 0.8f
#endif

#ifndef BARE_TOPP
#define BARE_TOPP 0.9f
#endif

#ifndef BARE_SEED
#define BARE_SEED 123456789ull
#endif

#ifndef BARE_HEAP_BYTES
#define BARE_HEAP_BYTES (22 * 1024 * 1024)
#endif
// 原本256MB的堆，现在改为24MB
#ifndef BARE_CPU_HZ
#define BARE_CPU_HZ CLOCK_FREQUENCY
#endif

#ifndef BARE_USE_CYCLE_COUNTER
// On some Spike/CRT setups the user-mode cycle CSR access is disabled and
// `rdcycle` traps (mcause=2). Default off for robustness; enable explicitly
// if your CRT config allows it.
#define BARE_USE_CYCLE_COUNTER 1
#endif

// ---------------------------------------------------------------------------
// Embedded binary symbols (provided by objcopy in the Makefile)
#ifndef MODEL_BIN_START
#error "MODEL_BIN_START is not defined. See Makefile rvbare instructions."
#endif
#ifndef MODEL_BIN_END
#error "MODEL_BIN_END is not defined. See Makefile rvbare instructions."
#endif
#ifndef TOKENIZER_BIN_START
#error "TOKENIZER_BIN_START is not defined. See Makefile rvbare instructions."
#endif
#ifndef TOKENIZER_BIN_END
#error "TOKENIZER_BIN_END is not defined. See Makefile rvbare instructions."
#endif

extern const unsigned char MODEL_BIN_START[];
extern const unsigned char MODEL_BIN_END[];
extern const unsigned char TOKENIZER_BIN_START[];
extern const unsigned char TOKENIZER_BIN_END[];
extern char _end[]; // Linker symbol for end of BSS

static inline size_t embedded_model_size(void) {
    return (size_t)(MODEL_BIN_END - MODEL_BIN_START);
}

static inline size_t embedded_tokenizer_size(void) {
    return (size_t)(TOKENIZER_BIN_END - TOKENIZER_BIN_START);
}

// ---------------------------------------------------------------------------
// Tiny bump allocator for bare-metal use
static unsigned char bare_heap[BARE_HEAP_BYTES];
static size_t bare_heap_offset = 0;

// ---------------------------------------------------------------------------
// Memory usage stats helpers
#if BARE_MEM_STATS
static uintptr_t initial_sp = 0;
#define STACK_FILL_PATTERN 0x55

static void paint_stack(void) {
    uintptr_t sp;
    asm volatile("mv %0, sp" : "=r"(sp));
    initial_sp = sp;
    
    // We assume stack grows down towards _end.
    // Safety gap of 4KB from _end
    uintptr_t stack_bottom = (uintptr_t)_end + 4096;
    
    // Paint if we have space
    if (sp > stack_bottom) {
        // Leave 256 bytes safety below current sp
        size_t len = (sp - 256) - stack_bottom;
        memset((void*)stack_bottom, STACK_FILL_PATTERN, len);
    }
}

static size_t get_stack_usage(void) {
    if (initial_sp == 0) return 0;
    uintptr_t stack_bottom = (uintptr_t)_end + 4096;
    unsigned char *p = (unsigned char *)stack_bottom;
    uintptr_t current_sp;
    asm volatile("mv %0, sp" : "=r"(current_sp));
    
    // Scan upwards for first non-pattern byte or reaching current sp
    // (limit scan to initial_sp to avoid overrun)
    while ((uintptr_t)p < initial_sp && *p == STACK_FILL_PATTERN) {
        p++;
    }
    return (size_t)(initial_sp - (uintptr_t)p);
}

static void print_memory_stats(void) {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("\r\n[bare] Memory Stats:\r\n");
    
    // Heap
    print_uart("  Heap Used:  ");
    print_uart_int_dec((unsigned long long)bare_heap_offset);
    print_uart(" / ");
    print_uart_int_dec((unsigned long long)BARE_HEAP_BYTES);
    print_uart(" bytes\r\n");
    
    // Stack
    size_t stack_used = get_stack_usage();
    print_uart("  Stack Used: ");
    print_uart_int_dec((unsigned long long)stack_used);
    print_uart(" bytes (approx)\r\n");
}
#endif

static void panic(const char *msg) {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    print_uart("[bare] PANIC: ");
    print_uart(msg);
    print_uart("\r\n");
    exit(1);
}

static void *bare_alloc(size_t size, size_t alignment) {
    if (alignment == 0) alignment = 8;
    size_t misalignment = bare_heap_offset % alignment;
    if (misalignment != 0) bare_heap_offset += alignment - misalignment;
    if (bare_heap_offset + size > BARE_HEAP_BYTES) panic("bare heap exhausted");
    void *ptr = &bare_heap[bare_heap_offset];
    bare_heap_offset += size;
    return ptr;
}

void *malloc(size_t size) {
    if (size == 0) size = 1;
    return bare_alloc(size, 8);
}

void *calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return bare_alloc(1, 8);
    size_t total = count * size;
    void *ptr = bare_alloc(total, 8);
    memset(ptr, 0, total);
    return ptr;
}

void free(void *ptr) { (void)ptr; }

// ---------------------------------------------------------------------------
#if BARE_USE_CYCLE_COUNTER
static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static inline long cycles_to_ms(uint64_t start, uint64_t end) {
#if BARE_CPU_HZ
    uint64_t delta = end - start;
    return (long)(delta / (BARE_CPU_HZ / 1000ULL));
#else
    (void)start; (void)end;
    return -1;
#endif
}
#else
static inline uint64_t rdcycle(void) { return 0; }
static inline long cycles_to_ms(uint64_t start, uint64_t end) {
    (void)start; (void)end;
    return -1;
}
#endif

static inline int is_printable(unsigned char c) {
    return (c >= 32 && c < 127);
}

static inline int is_whitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static inline int abs_int(int v) { return v < 0 ? -v : v; }

static inline float bare_fabsf(float x) { return x < 0.0f ? -x : x; }

float sqrtf(float x) {
    float xhalf = 0.5f * x;
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = 0x5f3759dfu - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - (xhalf * y * y));
    y = y * (1.5f - (xhalf * y * y));
    return x * y;
}

float expf(float x) {
    // Fast exp approximation based on 2^x with a short polynomial for mantissa
    const float LOG2E = 1.4426950408889634f;
    const float LN2 = 0.6931471805599453f;
    float y = x * LOG2E;
    float ip = y >= 0.0f ? (float)((int)(y + 0.5f)) : (float)((int)(y - 0.5f));
    float fp = y - ip;
    float poly = 1.0f + fp * (0.696065642f + fp * 0.224494337f);
    int exponent = (int)ip + 127;
    if (exponent <= 0) exponent = 0;
    if (exponent >= 255) exponent = 255;
    union { uint32_t i; float f; } bits;
    bits.i = (uint32_t)(exponent << 23);
    return bits.f * poly;
}

float sinf(float x) {
    const float PI = 3.1415926535897932f;
    const float TWO_PI = 6.2831853071795865f;
    while (x > PI) x -= TWO_PI;
    while (x < -PI) x += TWO_PI;
    const float B = 4.0f / PI;
    const float C = -4.0f / (PI * PI);
    float y = B * x + C * x * bare_fabsf(x);
    const float P = 0.225f;
    y = P * (y * bare_fabsf(y) - y) + y;
    return y;
}

float cosf(float x) {
    const float HALF_PI = 1.5707963267948966f;
    return sinf(x + HALF_PI);
}

// ---------------------------------------------------------------------------
// Transformer data structures (trimmed from run.c)
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;

    // Quantized copies of matmul weights (allocated in RAM at init)
    QuantizedTensor *wq_q;
    QuantizedTensor *wk_q;
    QuantizedTensor *wv_q;
    QuantizedTensor *wo_q;
    QuantizedTensor *w1_q;
    QuantizedTensor *w2_q;
    QuantizedTensor *w3_q;
    QuantizedTensor wcls_q;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = qt_alloc(p->dim);
    s->hq = qt_alloc(p->hidden_dim);
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        panic("RunState allocation failed");
    }
}

void free_run_state(RunState* s) { (void)s; }

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void init_transformer_from_embedded(Transformer *t) {
    size_t bytes = embedded_model_size();
    if (bytes < sizeof(Config)) panic("model blob too small");
    Config cfg_disk;
    memcpy(&cfg_disk, MODEL_BIN_START, sizeof(Config));
    int shared = cfg_disk.vocab_size > 0 ? 1 : 0;
    cfg_disk.vocab_size = abs_int(cfg_disk.vocab_size);
    t->config = cfg_disk;
    float *weights_ptr = (float*)(MODEL_BIN_START + sizeof(Config));
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared);

    // Quantize all matmul weights once at init so forward() can use int8 path.
    // This assumes the embedded checkpoint stores float weights (v1 format).
    {
        Config *p = &t->config;
        TransformerWeights *w = &t->weights;
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int hidden_dim = p->hidden_dim;

        w->wq_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->wk_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->wv_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->wo_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->w1_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->w2_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        w->w3_q = (QuantizedTensor*)calloc((size_t)p->n_layers, sizeof(QuantizedTensor));
        if (!w->wq_q || !w->wk_q || !w->wv_q || !w->wo_q || !w->w1_q || !w->w2_q || !w->w3_q) {
            panic("Quant weight table alloc failed");
        }

        for (int l = 0; l < p->n_layers; l++) {
            // wq: (dim, dim)
            {
                const int n = dim;
                const int d = dim;
                const int total = n * d;
                w->wq_q[l] = qt_alloc(total);
                quantize_const(&w->wq_q[l], w->wq + (size_t)l * (size_t)total, total);
            }
            // wk/wv: (kv_dim, dim)
            {
                const int n = dim;
                const int d = kv_dim;
                const int total = n * d;
                w->wk_q[l] = qt_alloc(total);
                w->wv_q[l] = qt_alloc(total);
                quantize_const(&w->wk_q[l], w->wk + (size_t)l * (size_t)total, total);
                quantize_const(&w->wv_q[l], w->wv + (size_t)l * (size_t)total, total);
            }
            // wo: (dim, dim)
            {
                const int n = dim;
                const int d = dim;
                const int total = n * d;
                w->wo_q[l] = qt_alloc(total);
                quantize_const(&w->wo_q[l], w->wo + (size_t)l * (size_t)total, total);
            }
            // w1/w3: (hidden_dim, dim)
            {
                const int n = dim;
                const int d = hidden_dim;
                const int total = n * d;
                w->w1_q[l] = qt_alloc(total);
                w->w3_q[l] = qt_alloc(total);
                quantize_const(&w->w1_q[l], w->w1 + (size_t)l * (size_t)total, total);
                quantize_const(&w->w3_q[l], w->w3 + (size_t)l * (size_t)total, total);
            }
            // w2: (dim, hidden_dim)
            {
                const int n = hidden_dim;
                const int d = dim;
                const int total = n * d;
                w->w2_q[l] = qt_alloc(total);
                quantize_const(&w->w2_q[l], w->w2 + (size_t)l * (size_t)total, total);
            }
        }

        // classifier: (vocab_size, dim)
        {
            const int n = dim;
            const int d = p->vocab_size;
            const int total = n * d;
            w->wcls_q = qt_alloc(total);
            quantize_const(&w->wcls_q, w->wcls, total);
        }
    }
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) { free_run_state(&t->state); (void)t; }

// ---------------------------------------------------------------------------
// Math helpers reused from run.c
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* row = w + i * n;
        for (int j = 0; j < n; j++) val += row[j] * x[j];
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    #if MATRIX_KERNEL_DEBUG_PRINT
    print_uart("[bare] enter foward...\r\n");
    #endif
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        #if MATRIX_KERNEL_DEBUG_PRINT
        print_uart("[bare] rmsnorm1...\r\n");
        #endif
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        int loff = l * p->seq_len * kv_dim;
        float* layer_key_cache = s->key_cache + loff;
        float* layer_val_cache = s->value_cache + loff;
        float* k_slot = layer_key_cache + pos * kv_dim;
        float* v_slot = layer_val_cache + pos * kv_dim;
        #if MATRIX_KERNEL_DEBUG_PRINT
        print_uart("[bare] quantize...\r\n");
        #endif
        quantize(&s->xq, s->xb, dim);
        #if MATRIX_KERNEL_DEBUG_PRINT
        print_uart("[bare] matmul_q...\r\n");
        #endif
        MATMUL_Q_TAGGED("wq", (int)l, s->q, &s->xq, w->wq_q + l, dim, dim);
        MATMUL_Q_TAGGED("wk", (int)l, s->k, &s->xq, w->wk_q + l, dim, kv_dim);
        MATMUL_Q_TAGGED("wv", (int)l, s->v, &s->xq, w->wv_q + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            float head_dim = (float)(i % head_size);
            float freq = expf(-9.210340371976184f * (head_dim / (float)head_size));
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float q0 = s->q[i];
            float q1 = s->q[i+1];
            s->q[i]   = q0 * fcr - q1 * fci;
            s->q[i+1] = q0 * fci + q1 * fcr;
            if (i < kv_dim) {
                float k0 = s->k[i];
                float k1 = s->k[i+1];
                s->k[i]   = k0 * fcr - k1 * fci;
                s->k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        memcpy(k_slot, s->k, kv_dim * sizeof(float));
        memcpy(v_slot, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            float scale = 1.0f / sqrtf((float)head_size);
            for (int t = 0; t <= pos; t++) {
                float* k = layer_key_cache + t * kv_dim + (h/kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                att[t] = score * scale;
            }
            for (int t = pos + 1; t < p->seq_len; t++) att[t] = -1e9f;
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            for (int i = 0; i < head_size; i++) xb[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float att_t = att[t];
                float* v = layer_val_cache + t * kv_dim + (h/kv_mul) * head_size;
                for (int i = 0; i < head_size; i++) xb[i] += att_t * v[i];
            }
            }

            quantize(&s->xq, s->xb, dim);
            MATMUL_Q_TAGGED("wo", (int)l, s->xb2, &s->xq, w->wo_q + l, dim, dim);
            for (int i = 0; i < dim; i++) x[i] = x[i] + s->xb2[i];

            rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
            quantize(&s->xq, s->xb, dim);
            MATMUL_Q_TAGGED("ffn", (int)l, s->hb, &s->xq, w->w1_q + l, dim, hidden_dim);
            MATMUL_Q_TAGGED("ffn", (int)l, s->hb2, &s->xq, w->w3_q + l, dim, hidden_dim);
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= 1.0f / (1.0f + expf(-val));
                s->hb[i] = val * s->hb2[i];
            }
            quantize(&s->hq, s->hb, hidden_dim);
            MATMUL_Q_TAGGED("ffn", (int)l, s->xb, &s->hq, w->w2_q + l, hidden_dim, dim);
            for (int i = 0; i < dim; i++) x[i] = x[i] + s->xb[i];
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        quantize(&s->xq, x, dim);
        MATMUL_Q_TAGGED("wcls", -1, s->logits, &s->xq, &w->wcls_q, p->dim, p->vocab_size);
        return s->logits;
    }

// ---------------------------------------------------------------------------
// Tokenizer (trimmed from run.c, adapted for embedded data)



typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static BARE_NO_AUTOVEC void swap_token(TokenIndex* a, TokenIndex* b) {
    TokenIndex tmp = *a; *a = *b; *b = tmp;
}

static BARE_NO_AUTOVEC void quicksort_tokens(TokenIndex* arr, int left, int right) {
    while (left < right) {
        int i = left;
        int j = right;
        char* pivot = arr[left + (right - left) / 2].str;
        while (i <= j) {
            while (strcmp(arr[i].str, pivot) < 0) i++;
            while (strcmp(arr[j].str, pivot) > 0) j--;
            if (i <= j) {
                swap_token(&arr[i], &arr[j]);
                i++; j--;
            }
        }
        if (left < j) quicksort_tokens(arr, left, j);
        left = i;
    }
}

static TokenIndex* tokenizer_sorted(Tokenizer* t) {
    if (t->sorted_vocab) return t->sorted_vocab;
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    if (!t->sorted_vocab) panic("token sorted alloc");
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    quicksort_tokens(t->sorted_vocab, 0, t->vocab_size - 1);
    return t->sorted_vocab;
}

static int token_lookup(Tokenizer* t, char *str) {
    TokenIndex* sorted = tokenizer_sorted(t);
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(sorted[mid].str, str);
        if (cmp == 0) return sorted[mid].id;
        if (cmp < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static int parse_byte_token(const char* piece, unsigned char* out) {
    if (!piece || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x') return 0;
    int hi = hex_nibble(piece[3]);
    int lo = hex_nibble(piece[4]);
    if (hi < 0 || lo < 0 || piece[5] != '>' || piece[6] != '\0') return 0;
    *out = (unsigned char)((hi << 4) | lo);
    return 1;
}

static void build_tokenizer_from_embedded(Tokenizer* t, int vocab_size) {
    t->vocab_size = vocab_size;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    const unsigned char *ptr = TOKENIZER_BIN_START;
    const unsigned char *end = TOKENIZER_BIN_END;
    if (ptr + sizeof(int) > end) panic("tokenizer blob too small");
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);
    t->vocab = malloc(vocab_size * sizeof(char*));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    if (!t->vocab || !t->vocab_scores) panic("tokenizer alloc");
    for (int i = 0; i < vocab_size; i++) {
        if (ptr + sizeof(float) + sizeof(int) > end) panic("tokenizer truncated");
        memcpy(t->vocab_scores + i, ptr, sizeof(float));
        ptr += sizeof(float);
        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);
        if (ptr + len > end) panic("token string truncated");
        t->vocab[i] = malloc(len + 1);
        if (!t->vocab[i]) panic("token malloc");
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }
}

void free_tokenizer(Tokenizer* t) { (void)t; }

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (parse_byte_token(piece, &byte_val)) piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

static void concat_tokens(char *dst, size_t cap, const char *a, const char *b) {
    size_t idx = 0;
    for (size_t i = 0; a[i] && idx < cap - 1; i++) dst[idx++] = a[i];
    for (size_t i = 0; b[i] && idx < cap - 1; i++) dst[idx++] = b[i];
    dst[idx] = '\0';
}

static int str_lookup(Tokenizer* t, char *str) {
    return token_lookup(t, str);
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    #if MATRIX_KERNEL_DEBUG_PRINT
    print_uart("[bare] enter encode...\r\n");
    #endif

    if (text == NULL) panic("encode text NULL");
    char* str_buffer = malloc((t->max_token_length*2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(t, " ");
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(t, str_buffer);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                char buf[2] = { str_buffer[i], '\0' };
                int single_id = str_lookup(t, buf);
                if (single_id == -1) panic("tokenizer single missing");
                tokens[(*n_tokens)++] = single_id;
            }
        }
        str_len = 0;
    }
    char merge_buf[1024];
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            concat_tokens(merge_buf, sizeof(merge_buf), t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(t, merge_buf);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

void safe_printf(char *piece) {
    if (!piece || !piece[0]) return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(is_printable(byte_val) || is_whitespace(byte_val))) return;
    }
    print_uart(piece);
}

// ---------------------------------------------------------------------------
// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_p = probabilities[i];
            max_i = i;
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = seed;
    sampler->probindex = malloc(vocab_size * sizeof(ProbIndex));
    if (!sampler->probindex) panic("sampler alloc");
}

void free_sampler(Sampler* sampler) { free(sampler->probindex); }

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].prob = probabilities[i];
            probindex[n0].index = i;
            n0++;
        }
    }
    for (int i = 1; i < n0; i++) {
        ProbIndex key = probindex[i];
        int j = i - 1;
        while (j >= 0 && probindex[j].prob < key.prob) {
            probindex[j+1] = probindex[j];
            j--;
        }
        probindex[j+1] = key;
    }
    float cumulative = 0.0f;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += probindex[i].prob;
        if (cumulative > topp) { last = i; break; }
    }
    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last].index;
}

int sample_token(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) logits[q] /= sampler->temperature;
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ---------------------------------------------------------------------------
// Batched Matmul for Prefill
// ---------------------------------------------------------------------------

// Reference implementation (Scalar) ported from test_matrix_kernel_batch.c
static void matmul_ref(float *out, const int8_t *xq, const float *xs, 
                       const int8_t *wq, const float *ws, 
                       int n, int d, int gs, int batch) {
    int num_groups = (n + gs - 1) / gs;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < d; i++) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) {
                int group = j / gs;
                int8_t w_val = wq[i * n + j];
                float w_scale = ws[i * num_groups + group];
                int8_t x_val = xq[b * n + j];
                float x_scale = xs[b * num_groups + group];
                acc += ((float)w_val * w_scale) * ((float)x_val * x_scale);
            }
            out[b * d + i] = acc;
        }
    }
}

// HW Accelerated Batched Matmul
static void matmul_q_batch(float* xout, const int8_t* xq, const float* xs, 
                                          const int8_t* wq, const float* ws, 
                                          int n, int d, int batch) {
    // Currently using scalar reference for verification
    matmul_ref(xout, xq, xs, wq, ws, n, d, GS, batch);
}

// Batched quantization for activations during prefill
void quantize_batch(int8_t* xq, float* xs, float* x, int n, int batch) {
    const int num_groups = (n + GS - 1) / GS;
    const float Q_MAX = 127.0f;

    for (int b = 0; b < batch; b++) {
        float* x_row = x + b * n;
        int8_t* xq_row = xq + b * n;
        float* xs_row = xs + b * num_groups;
        
        for (int group = 0; group < num_groups; group++) {
            const int base = group * GS;
            const int count = (base + GS <= n) ? GS : (n - base);
            float xmax = 0.0f;
            for (int i = 0; i < count; i++) {
                float v = bare_fabsf(x_row[base + i]);
                if (v > xmax) xmax = v;
            }
            float scale = (xmax > 0.0f) ? (xmax / Q_MAX) : 1.0f;
            xs_row[group] = scale;
            if (xmax == 0.0f) {
                for (int i = 0; i < count; i++) xq_row[base + i] = 0;
            } else {
                float inv = 1.0f / scale;
                for (int i = 0; i < count; i++) {
                    xq_row[base + i] = q_round_clamp_i8(x_row[base + i] * inv);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer Prefill (Matrix-Matrix Optimization)
// ---------------------------------------------------------------------------

void transformer_prefill(Transformer *t, int* tokens, int n_tokens) {
    if (n_tokens <= 0) return;
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    const int num_groups_x = (dim + GS - 1) / GS;
    const int num_groups_h = (hidden_dim + GS - 1) / GS;

#if MATRIX_KERNEL_DEBUG_PRINT
    print_uart("[bare] prefill batch size: ");
    print_uart_int_dec((uint64_t)n_tokens);
    print_uart("\r\n");
#endif

    // Allocate batch buffers from heap (Bump allocator). 
    // Uses the large 256MB bare_heap defined earlier.
    float* x_batch = (float*)malloc(n_tokens * dim * sizeof(float));
    float* xb_batch = (float*)malloc(n_tokens * dim * sizeof(float));
    float* xb2_batch = (float*)malloc(n_tokens * dim * sizeof(float));
    float* q_batch = (float*)malloc(n_tokens * dim * sizeof(float));
    float* k_batch = (float*)malloc(n_tokens * kv_dim * sizeof(float));
    float* v_batch = (float*)malloc(n_tokens * kv_dim * sizeof(float));
    float* hb_batch = (float*)malloc(n_tokens * hidden_dim * sizeof(float));
    float* hb2_batch = (float*)malloc(n_tokens * hidden_dim * sizeof(float));
    
    int8_t* xq_batch_q = (int8_t*)malloc(n_tokens * dim * sizeof(int8_t));
    float* xq_batch_s = (float*)malloc(n_tokens * num_groups_x * sizeof(float));
    int8_t* hq_batch_q = (int8_t*)malloc(n_tokens * hidden_dim * sizeof(int8_t));
    float* hq_batch_s = (float*)malloc(n_tokens * num_groups_h * sizeof(float));

    // Embedding lookup
    for (int i = 0; i < n_tokens; i++) {
        memcpy(x_batch + (size_t)i * dim, w->token_embedding_table + (size_t)tokens[i] * dim, dim * sizeof(float));
    }

    for (int l = 0; l < p->n_layers; l++) {
        // RMSNorm (Att)
        for (int i = 0; i < n_tokens; i++) {
            rmsnorm(xb_batch + i * dim, x_batch + i * dim, w->rms_att_weight + l*dim, dim);
        }

        // QKV Matmuls (Batched Matrix-Matrix)
        quantize_batch(xq_batch_q, xq_batch_s, xb_batch, dim, n_tokens);
        matmul_q_batch(q_batch, xq_batch_q, xq_batch_s, w->wq_q[l].q, w->wq_q[l].s, dim, dim, n_tokens);
        matmul_q_batch(k_batch, xq_batch_q, xq_batch_s, w->wk_q[l].q, w->wk_q[l].s, dim, kv_dim, n_tokens);
        matmul_q_batch(v_batch, xq_batch_q, xq_batch_s, w->wv_q[l].q, w->wv_q[l].s, dim, kv_dim, n_tokens);

        // RoPE and KV Cache Update (Position-dependent)
        for (int i = 0; i < n_tokens; i++) {
            int pos = i;
            float* q = q_batch + i * dim;
            float* k = k_batch + i * kv_dim;
            for (int j = 0; j < dim; j += 2) {
                float h_dim = (float)(j % head_size);
                float freq = expf(-9.210340371976184f * (h_dim / (float)head_size));
                float val = (float)pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                float q0 = q[j];   float q1 = q[j+1];
                q[j]   = q0 * fcr - q1 * fci;
                q[j+1] = q0 * fci + q1 * fcr;
                if (j < kv_dim) {
                    float k0 = k[j]; float k1 = k[j+1];
                    k[j]   = k0 * fcr - k1 * fci;
                    k[j+1] = k0 * fci + k1 * fcr;
                }
            }
            // Populate Cache
            float* layer_k_cache = s->key_cache + (l * p->seq_len * kv_dim);
            float* layer_v_cache = s->value_cache + (l * p->seq_len * kv_dim);
            memcpy(layer_k_cache + pos * kv_dim, k, kv_dim * sizeof(float));
            memcpy(layer_v_cache + pos * kv_dim, v_batch + i * kv_dim, kv_dim * sizeof(float));
        }

        // Causal Attention
        for (int i = 0; i < n_tokens; i++) {
            int pos = i;
            float* layer_k_cache = s->key_cache + (l * p->seq_len * kv_dim);
            float* layer_v_cache = s->value_cache + (l * p->seq_len * kv_dim);
            
            for (int h = 0; h < p->n_heads; h++) {
                float* q = q_batch + i * dim + h * head_size;
                float* att = s->att + h * p->seq_len;
                float scale = 1.0f / sqrtf((float)head_size);
                for (int t = 0; t <= pos; t++) {
                    float* k = layer_k_cache + t * kv_dim + (h/kv_mul) * head_size;
                    float score = 0.0f;
                    for (int n = 0; n < head_size; n++) score += q[n] * k[n];
                    att[t] = score * scale;
                }
                for (int t = pos + 1; t < p->seq_len; t++) att[t] = -1e9f;
                softmax(att, pos + 1);
                float* xb_out = xb_batch + i * dim + h * head_size;
                for (int n = 0; n < head_size; n++) xb_out[n] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    float att_t = att[t];
                    float* v = layer_v_cache + t * kv_dim + (h/kv_mul) * head_size;
                    for (int n = 0; n < head_size; n++) xb_out[n] += att_t * v[n];
                }
            }
        }

        // Output Matmul (Batched)
        quantize_batch(xq_batch_q, xq_batch_s, xb_batch, dim, n_tokens);
        matmul_q_batch(xb2_batch, xq_batch_q, xq_batch_s, w->wo_q[l].q, w->wo_q[l].s, dim, dim, n_tokens);
        for(int i=0; i<n_tokens*dim; i++) x_batch[i] += xb2_batch[i];

        // FFN (Batched)
        for (int i = 0; i < n_tokens; i++) {
            rmsnorm(xb_batch + i * dim, x_batch + i * dim, w->rms_ffn_weight + l*dim, dim);
        }
        quantize_batch(xq_batch_q, xq_batch_s, xb_batch, dim, n_tokens);
        matmul_q_batch(hb_batch, xq_batch_q, xq_batch_s, w->w1_q[l].q, w->w1_q[l].s, dim, hidden_dim, n_tokens);
        matmul_q_batch(hb2_batch, xq_batch_q, xq_batch_s, w->w3_q[l].q, w->w3_q[l].s, dim, hidden_dim, n_tokens);
        
        for (int i = 0; i < n_tokens * hidden_dim; i++) {
            float val = hb_batch[i];
            val *= 1.0f / (1.0f + expf(-val));
            hb_batch[i] = val * hb2_batch[i];
        }
        
        quantize_batch(hq_batch_q, hq_batch_s, hb_batch, hidden_dim, n_tokens);
        matmul_q_batch(xb_batch, hq_batch_q, hq_batch_s, w->w2_q[l].q, w->w2_q[l].s, hidden_dim, dim, n_tokens);
        for(int i=0; i<n_tokens*dim; i++) x_batch[i] += xb_batch[i];
    }
    
    // Resume point: Sync current state to the last prefilled token's output
    memcpy(s->x, x_batch + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
}

// ---------------------------------------------------------------------------
// Generation loop (prompt is compile-time string)
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
#if MATRIX_KERNEL_DEBUG_PRINT
    print_uart("[bare] enter generate...\r\n");
#endif
    if (prompt == NULL) panic("prompt NULL");
    int num_prompt_tokens = 0;
    int* prompt_tokens = malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) panic("prompt encode empty");

#if BARE_USE_CYCLE_COUNTER
    uint64_t start_cycles = 0;
    uint64_t ttft_start = rdcycle();
    uint64_t ttft_end = 0;
#endif
    int token = prompt_tokens[0];
    int pos = 0;

    // Prefill Optimization: Process all prompt tokens except the last one in a single batch
    // using the Matrix-Matrix multiplication optimization.
    if (num_prompt_tokens > 1) {
        transformer_prefill(transformer, prompt_tokens, num_prompt_tokens - 1);
        // add print
        for (int i = 0; i < num_prompt_tokens - 1; i++) {
            int p_token = prompt_tokens[i];
            int prev_p_token = (i == 0) ? 1 : prompt_tokens[i-1];
            char* piece = decode(tokenizer, prev_p_token, p_token);
            safe_printf(piece); 
        }
        pos = num_prompt_tokens - 1;
        token = prompt_tokens[pos];
    }

    int line_char_count = 0;
    char word_buffer[256];
    int word_len = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_token(sampler, logits);
            #if BARE_USE_CYCLE_COUNTER
            if (ttft_end == 0) ttft_end = rdcycle();
            #endif
        }
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        //safe_printf(piece);
        if (piece) {
            int should_print = 1;
            if (piece[0] == '\0') should_print = 0;
            else if (piece[1] == '\0') {
                unsigned char byte_val = piece[0];
                if (!(is_printable(byte_val) || is_whitespace(byte_val))) should_print = 0;
            }

            if (should_print) {
                for (char *c = piece; *c != '\0'; c++) {
                    if (is_whitespace((unsigned char)*c)) {
                        if (word_len > 0) {
                            if (line_char_count + word_len > 80) {
                                print_uart("\r\n");
                                line_char_count = 0;
                            }
                            for (int i = 0; i < word_len; i++) write_serial((uint8_t)word_buffer[i]);
                            line_char_count += word_len;
                            word_len = 0;
                        }
                        if (*c == '\n' || *c == '\r') {
                            print_uart("\r\n");
                            line_char_count = 0;
                        } else {
                            if (line_char_count >= 80) {
                                print_uart("\r\n");
                                line_char_count = 0;
                            } else {
                                write_serial((uint8_t)*c);
                                line_char_count++;
                            }
                        }
                    } else {
                        if (word_len < 255) {
                            word_buffer[word_len++] = *c;
                        } else {
                            if (line_char_count + word_len > 80) {
                                print_uart("\r\n");
                                line_char_count = 0;
                            }
                            for (int i = 0; i < word_len; i++) write_serial((uint8_t)word_buffer[i]);
                            line_char_count += word_len;
                            word_len = 0;
                            word_buffer[word_len++] = *c;
                        }
                    }
                }
            }
        }
            #if BARE_MEM_STATS
            print_memory_stats();
            #endif
        token = next;
#if BARE_USE_CYCLE_COUNTER
        if (start_cycles == 0) start_cycles = rdcycle();
#endif
    }
    print_uart("\r\n");
#if BARE_USE_CYCLE_COUNTER
    if (start_cycles != 0) {
        uint64_t end_cycles = rdcycle();
        long ms = cycles_to_ms(start_cycles, end_cycles);
        if (ms > 0 && pos > 1) {
            float tok_s = (float)(pos-1) / (ms / 1000.0f);
            print_uart("[bare] tok/s: ");
            print_float_fixed3(tok_s);
            print_uart("\r\n");
        } else {
            print_uart("[bare] cycles: ");
            print_uart_int_dec((uint64_t)(end_cycles - start_cycles));
            print_uart("\r\n");
        }
    }
    if (ttft_end != 0) {
        long ttft_ms = cycles_to_ms(ttft_start, ttft_end);
        print_uart("[bare] TTFT: ");
        print_uart_int_dec((uint64_t)ttft_ms);
        print_uart(" ms\r\n");
    }
#endif
    free(prompt_tokens);
}

// ---------------------------------------------------------------------------
int main(void) {
    #if BARE_MEM_STATS
    paint_stack();
    #endif
    enable_rvv_state();
// #if BARE_USE_CYCLE_COUNTER
//     g_bare_boot_cycles = rdcycle();
// #endif

    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    #if BARE_MEM_STATS
    print_memory_stats();
    #endif
    print_uart("[bare] model bytes: ");
    print_uart_int_dec((uint64_t)embedded_model_size());
    print_uart("\r\n");
    print_uart("[bare] tokenizer bytes: ");
    print_uart_int_dec((uint64_t)embedded_tokenizer_size());
    print_uart("\r\n");

    print_uart("[bare] init_transformer_from_embedded...\r\n");
    Transformer transformer;
    init_transformer_from_embedded(&transformer);

    print_uart("[bare] build_tokenizer_from_embedded...\r\n");
    Tokenizer tokenizer;
    build_tokenizer_from_embedded(&tokenizer, transformer.config.vocab_size);

    print_uart("[bare] build_sampler...\r\n");
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, BARE_TEMPERATURE, BARE_TOPP, BARE_SEED);

    print_uart("[bare] generate...\r\n");
    static char prompt[] = BARE_PROMPT;
    int steps = BARE_STEPS;
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    #if BARE_MEM_STATS
    print_memory_stats();
    #endif

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    #if BARE_MEM_STATS
    print_memory_stats();
    #endif
    print_uart("[bare] done.\r\n");
    return 0;
}
