// Bare-metal friendly inference harness for llama2.c
// ---------------------------------------------------
// This variant removes all host OS dependencies so the model can execute in a
// freestanding Spike tohost/fromhost environment. The model and tokenizer
// binaries are embedded directly into the ELF via objcopy; see the Makefile
// rvbare target for how MODEL_BIN/TOKENIZER_BIN become linker symbols used here.

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <riscv_vector.h>
#include "riscv_vector_kernels.h"

// ---------------------------------------------------------------------------
// Tiny libc hooks provided by bare_syscalls.c (no system headers required)
typedef long ssize_t;

void *memcpy(void *dest, const void *src, size_t len);
void *memset(void *dest, int byte, size_t len);
int printf(const char *fmt, ...);
void exit(int code);
int strcmp(const char *s1, const char *s2);
size_t strlen(const char *s);

// ---------------------------------------------------------------------------
// Build-time knobs (override via e.g. `make rvbare CFLAGS+=-DBARE_STEPS=128`)
#ifndef BARE_PROMPT
#define BARE_PROMPT "what is math?" 
#endif
//"Once upon a time"
#ifndef BARE_STEPS
#define BARE_STEPS 512
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
#define BARE_HEAP_BYTES (32 * 1024 * 1024)
#endif

#ifndef BARE_CPU_HZ
#define BARE_CPU_HZ 0
#endif

#ifndef BARE_USE_CYCLE_COUNTER
#define BARE_USE_CYCLE_COUNTER 0
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

static void panic(const char *msg) {
    printf("[bare] PANIC: %s\n", msg);
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
    return __builtin_sqrtf(x);
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
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
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

// softmax implementation moved to riscv_vector_kernels.h

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* row = w + i * n;
        for (int j = 0; j < n; j++) val += row[j] * x[j];
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
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
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        int loff = l * p->seq_len * kv_dim;
        float* layer_key_cache = s->key_cache + loff;
        float* layer_val_cache = s->value_cache + loff;
        float* k_slot = layer_key_cache + pos * kv_dim;
        float* v_slot = layer_val_cache + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

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

            matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
            for (int i = 0; i < dim; i++) x[i] = x[i] + s->xb2[i];

            rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
            matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= 1.0f / (1.0f + expf(-val));
                s->hb[i] = val * s->hb2[i];
            }
            matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
            for (int i = 0; i < dim; i++) x[i] = x[i] + s->xb[i];
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
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

static void swap_token(TokenIndex* a, TokenIndex* b) {
    TokenIndex tmp = *a; *a = *b; *b = tmp;
}

static void quicksort_tokens(TokenIndex* arr, int left, int right) {
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
    printf("%s", piece);
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
// Generation loop (prompt is compile-time string)
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    if (prompt == NULL) panic("prompt NULL");
    int num_prompt_tokens = 0;
    int* prompt_tokens = malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) panic("prompt encode empty");

#if BARE_USE_CYCLE_COUNTER
    uint64_t start_cycles = 0;
#endif
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_token(sampler, logits);
        }
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;
#if BARE_USE_CYCLE_COUNTER
        if (start_cycles == 0) start_cycles = rdcycle();
#endif
    }
    printf("\n");
#if BARE_USE_CYCLE_COUNTER
    if (start_cycles != 0) {
        uint64_t end_cycles = rdcycle();
        long ms = cycles_to_ms(start_cycles, end_cycles);
        if (ms > 0 && pos > 1) {
            float tok_s = (float)(pos-1) / (ms / 1000.0f);
            printf("[bare] tok/s: %f\n", tok_s);
        } else {
            printf("[bare] cycles: %llu\n", (unsigned long long)(end_cycles - start_cycles));
        }
    }
#endif
    free(prompt_tokens);
}

// ---------------------------------------------------------------------------
int main(void) {
    // Enable VS extension (bits 9-10 of mstatus)
    // VS: 0 = Off, 1 = Initial, 2 = Clean, 3 = Dirty
    // We set it to 1 (Initial) -> 0x200
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x200; 
    asm volatile("csrw mstatus, %0" :: "r"(mstatus));

    printf("[bare] model bytes: %llu\n", (unsigned long long)embedded_model_size());
    printf("[bare] tokenizer bytes: %llu\n", (unsigned long long)embedded_tokenizer_size());

    Transformer transformer;
    init_transformer_from_embedded(&transformer);

    Tokenizer tokenizer;
    build_tokenizer_from_embedded(&tokenizer, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, BARE_TEMPERATURE, BARE_TOPP, BARE_SEED);

    static char prompt[] = BARE_PROMPT;
    int steps = BARE_STEPS;
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    printf("[bare] done.\n");
    return 0;
}
