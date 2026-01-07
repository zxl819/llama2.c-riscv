#ifndef MATRIX_KERNEL_H
#define MATRIX_KERNEL_H

// Tokenizer (trimmed from run.c, adapted for embedded data)
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

#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "uart_helper.c"

// ---------------------------------------------------------------------------
// Optional debug printing (UART/printf via uart_helper.c)
// Enable at build time with: -DMATRIX_KERNEL_DEBUG_PRINT=1
// You can also reduce spam with: -DMATRIX_KERNEL_DEBUG_PRINT_EVERY=128

#ifndef MATRIX_KERNEL_DEBUG_PRINT
#define MATRIX_KERNEL_DEBUG_PRINT 1
#endif

#ifndef MATRIX_KERNEL_DEBUG_PRINT_EVERY
#define MATRIX_KERNEL_DEBUG_PRINT_EVERY 1
#endif

// Verbose debug: prints intermediate variables/partials inside the kernel.
// Enable with: -DMATRIX_KERNEL_DEBUG_PRINT=1 -DMATRIX_KERNEL_DEBUG_VERBOSE=1
// Use MATRIX_KERNEL_DEBUG_PRINT_EVERY to rate-limit (call-based).
#ifndef MATRIX_KERNEL_DEBUG_VERBOSE
#define MATRIX_KERNEL_DEBUG_VERBOSE 1
#endif

/* Optional: print debug only for a specific i0 block (row start). Set to
   -1 to disable. For targeted debugging set to 32 to print only rows 32-63. */
#ifndef MATRIX_KERNEL_DEBUG_PRINT_I0
#define MATRIX_KERNEL_DEBUG_PRINT_I0 -1
#endif

// For verbose array dumps, print at most this many elements per array.
// This is purely to limit UART output volume (the remainder prints as "...").
#ifndef MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS
#define MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS 16
#endif

#if MATRIX_KERNEL_DEBUG_PRINT
// Debug context (layer + matrix/op name). This is intentionally lightweight
// and header-only, so each translation unit that includes this header gets its
// own context storage.
static const char *g_matrix_kernel_debug_name = "?";
static int g_matrix_kernel_debug_layer = -1;

static inline void matrix_kernel_debug_set_context(const char *name, int layer) {
    g_matrix_kernel_debug_name = name ? name : "?";
    g_matrix_kernel_debug_layer = layer;
}

#ifndef MATRIX_KERNEL_DEBUG_SET_CONTEXT
// Usage: MATRIX_KERNEL_DEBUG_SET_CONTEXT("wq", layer)
#define MATRIX_KERNEL_DEBUG_SET_CONTEXT(name, layer) matrix_kernel_debug_set_context((name), (layer))
#endif

static inline void matrix_kernel_debug_print_u64(uint64_t v) {
    // Implemented in uart_helper.c; prints uint64 in decimal.
    print_uart_int_dec(v);
}

static inline void matrix_kernel_debug_print_hex_u64(uint64_t v) {
    print_uart("0x");
    // Implemented in uart_helper.c; prints 16 hex digits.
    print_uart_hex((unsigned long long)v);
}

static inline void matrix_kernel_debug_print_ptr(const void *p) {
    matrix_kernel_debug_print_hex_u64((uint64_t)(uintptr_t)p);
}

static inline void matrix_kernel_debug_print_f32(float f) {
    // Implemented in uart_helper.c; prints float in fixed format.
    print_float_fixed3(f);
}

static inline void matrix_kernel_debug_print_i32(int32_t v) {
    // Implemented in uart_helper.c; prints signed int32 in decimal.
    print_dec32(v);
}

static inline void matrix_kernel_debug_print_i8(int8_t v) {
    matrix_kernel_debug_print_i32((int32_t)v);
}

static inline void matrix_kernel_debug_print_arr_i8(const char *label, const int8_t *a, int n) {
    print_uart(label);
    print_uart(" n=");
    matrix_kernel_debug_print_u64((uint64_t)n);
    print_uart(" vals=");
    const int lim = (n < MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS) ? n : MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS;
    for (int i = 0; i < lim; i++) {
        if ((i % 8) == 0) {
            if (i != 0) {
                print_uart("\r\n");
                print_uart("    ");
            }
        } else {
            write_serial((uint8_t)' ');
        }
        matrix_kernel_debug_print_i8(a[i]);
    }
    if (n > lim) print_uart(" ...");
    print_uart("\r\n");
}

static inline void matrix_kernel_debug_print_arr_i32(const char *label, const int32_t *a, int n) {
    print_uart(label);
    print_uart(" n=");
    matrix_kernel_debug_print_u64((uint64_t)n);
    print_uart(" vals=");
    const int lim = (n < MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS) ? n : MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS;
    for (int i = 0; i < lim; i++) {
        if ((i % 8) == 0) {
            if (i != 0) {
                print_uart("\r\n");
                print_uart("    ");
            }
        } else {
            write_serial((uint8_t)' ');
        }
        matrix_kernel_debug_print_i32(a[i]);
    }
    if (n > lim) print_uart(" ...");
    print_uart("\r\n");
}

static inline void matrix_kernel_debug_print_arr_f32(const char *label, const float *a, int n) {
    print_uart(label);
    print_uart(" n=");
    matrix_kernel_debug_print_u64((uint64_t)n);
    print_uart(" vals=");
    const int lim = (n < MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS) ? n : MATRIX_KERNEL_DEBUG_VERBOSE_MAX_ELEMS;
    for (int i = 0; i < lim; i++) {
        if ((i % 8) == 0) {
            if (i != 0) {
                print_uart("\r\n");
                print_uart("    ");
            }
        } else {
            write_serial((uint8_t)' ');
        }
        matrix_kernel_debug_print_f32(a[i]);
    }
    if (n > lim) print_uart(" ...");
    print_uart("\r\n");
}

static inline void matrix_kernel_debug_print_i64(int64_t v) {
    if (v < 0) {
        write_serial((uint8_t)'-');
        matrix_kernel_debug_print_u64((uint64_t)(-v));
        return;
    }
    matrix_kernel_debug_print_u64((uint64_t)v);
}

static inline void matrix_kernel_debug_print_prefix(void) {
    print_uart("[mk]");
    print_uart("[l=");
    matrix_kernel_debug_print_i64((int64_t)g_matrix_kernel_debug_layer);
    print_uart("]");
    print_uart("[");
    print_uart(g_matrix_kernel_debug_name);
    print_uart("] ");
}

static inline void matrix_kernel_debug_print_pair(const char *label, uint64_t v) {
    print_uart(label);
    matrix_kernel_debug_print_u64(v);
}
#endif

static inline void debug_delay_cycles(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) {
        asm volatile("nop");
    }
}

// Global constants and static buffers to avoid stack usage for pack buffers
enum { MK_MMAX = 32 };
static int32_t c_pack_mat[MK_MMAX * 32] __attribute__((aligned(64)));
static int32_t c_pack[MK_MMAX] __attribute__((aligned(64)));

/* Static scratch buffer to materialize strided ws into a contiguous vector.
    Kept global/static to avoid stack usage on bare-metal. */
extern float matrix_kernel_ws_pack_buf[MK_MMAX] __attribute__((aligned(64)));

/* Sub-buffer used by the blocked wrapper to hold per-row ws for an n-chunk */
static float matrix_kernel_ws_sub_buf[MK_MMAX * MK_MMAX] __attribute__((aligned(64)));

/* Static scratch buffers to avoid stack usage for packed A/B in qmatmul */
static int8_t matrix_kernel_a_pack_buf[MK_MMAX * 128] __attribute__((aligned(64)));
static int8_t matrix_kernel_b_pack_buf[32 * 128] __attribute__((aligned(64))); 

// Matrix extension i8 x i8 -> i32 accumulation kernel.
// Uses raw encodings (".word") so it can assemble under toolchains
// that don't recognize the Matrix mnemonics.
//
// Computes: C[m,n] += A[m,k] * B[n,k]^T  (i.e., B is provided as (n x k) row-major)
// A: row-major (m x k)
// B: row-major (n x k)  (each row is a vector of length k)
// C: row-major (m x n) int32



static BARE_NO_AUTOVEC inline int matrix_kernel_msettilem(int rem) {
    register size_t a0 asm("a0") = (size_t)rem;
    asm volatile(".word 0x04055577\n\t# msettilem a0,a0" : "+r"(a0) :: "memory");
    return (int)a0;
}

static BARE_NO_AUTOVEC inline int matrix_kernel_msettilen(int rem) {
    register size_t a0 asm("a0") = (size_t)rem;
    asm volatile(".word 0x04054577\n\t# msettilen a0,a0" : "+r"(a0) :: "memory");
    return (int)a0;
}

static BARE_NO_AUTOVEC inline int matrix_kernel_msettilek(int rem) {
    register size_t a0 asm("a0") = (size_t)rem;
    asm volatile(".word 0x04056577\n\t# msettilek a0,a0" : "+r"(a0) :: "memory");
    return (int)a0;
}

static BARE_NO_AUTOVEC inline void matrix_kernel_mlce32_acc0(const int32_t *base, int stride_bytes) {
    register const int32_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    asm volatile(".word 0x00b52077\n\t# mlce32.m acc0,(a0),a1" :: "r"(a0), "r"(a1) : "memory");
}

static BARE_NO_AUTOVEC inline void matrix_kernel_msce32_acc0(int32_t *base, int stride_bytes) {
    register int32_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    asm volatile(".word 0x02b52077\n\t# msce32.m acc0,(a0),a1" :: "r"(a0), "r"(a1) : "memory");
}

static BARE_NO_AUTOVEC inline void matrix_kernel_mlae8_tr0(const int8_t *base, int stride_bytes) {
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    asm volatile(".word 0x04b50077\n\t# mlae8.m tr0,(a0),a1" :: "r"(a0), "r"(a1) : "memory");
}

static BARE_NO_AUTOVEC inline void matrix_kernel_mlbe8_tr1(const int8_t *base, int stride_bytes) {
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    asm volatile(".word 0x08b500f7\n\t# mlbe8.m tr1,(a0),a1" :: "r"(a0), "r"(a1) : "memory");
}

static BARE_NO_AUTOVEC inline void matrix_kernel_mqma_b_acc0_tr0_tr1(void) {
    asm volatile(".word 0x28180877\n\t# mqma.b.mm acc0,tr0,tr1" ::: "memory");
}

extern int matrix_kernel_matmul_i8_i32_abt(const int8_t *A, const int8_t *B, int32_t *C,
                                                  int m, int n, int k);

// ---------------------------------------------------------------------------
// Strided variant: allows A/B/C row stride to be specified explicitly (bytes).
// This is useful to avoid scalar "packing" loops when the source tensors are
// already laid out row-major but with a row stride != k.
//
// A: base pointer to row 0, row stride = lda_bytes
// B: base pointer to row 0, row stride = ldb_bytes
// C: base pointer to row 0, row stride = ldc_bytes
static BARE_NO_AUTOVEC inline int matrix_kernel_matmul_i8_i32_abt_strided(const int8_t *A, int lda_bytes,
                                                          const int8_t *B, int ldb_bytes,
                                                          int32_t *C, int ldc_bytes,
                                                          int m, int n, int k) {
    int tile_m = 0, tile_n = 0, tile_k = 0;
    //int tile_m_pad = 0, tile_n_pad = 0, tile_k_pad = 0;

    for (int i = 0; i < m; i += tile_m) {
        tile_m = matrix_kernel_msettilem(m - i);
        //tile_m_pad = matrix_kernel_msettilem(8);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = matrix_kernel_msettilen(n - j);
            //tile_n_pad = matrix_kernel_msettilen(8);
            int32_t *cptr = (int32_t *)((uint8_t *)C + (size_t)i * (size_t)ldc_bytes) + j;

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            static uint64_t s_strided_calls = 0;
            const uint64_t dbg_id = ++s_strided_calls;
            if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
                matrix_kernel_debug_print_prefix();
                print_uart("abt_strided id=");
                matrix_kernel_debug_print_u64(dbg_id);
                print_uart(" m=");
                matrix_kernel_debug_print_u64((uint64_t)m);
                print_uart(" n=");
                matrix_kernel_debug_print_u64((uint64_t)n);
                print_uart(" k=");
                matrix_kernel_debug_print_u64((uint64_t)k);
                print_uart(" lda=");
                matrix_kernel_debug_print_u64((uint64_t)lda_bytes);
                print_uart(" ldb=");
                matrix_kernel_debug_print_u64((uint64_t)ldb_bytes);
                print_uart(" ldc=");
                matrix_kernel_debug_print_u64((uint64_t)ldc_bytes);
                print_uart("\r\n");

                matrix_kernel_debug_print_prefix();
                print_uart(" i=");
                matrix_kernel_debug_print_u64((uint64_t)i);
                print_uart(" tile_m=");
                matrix_kernel_debug_print_u64((uint64_t)tile_m);
                print_uart(" j=");
                matrix_kernel_debug_print_u64((uint64_t)j);
                print_uart(" tile_n=");
                matrix_kernel_debug_print_u64((uint64_t)tile_n);
                print_uart(" cptr=");
                matrix_kernel_debug_print_ptr(cptr);
                print_uart("\r\n");
            }
#endif
            matrix_kernel_mlce32_acc0((const int32_t *)cptr, ldc_bytes);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = matrix_kernel_msettilek(k - kk);
                //tile_k_pad = matrix_kernel_msettilek(8);
                const int8_t *aptr = (const int8_t *)((const uint8_t *)A + (size_t)i * (size_t)lda_bytes) + kk;
                // B is provided as (n x k) row-major, with explicit row stride.
                const int8_t *bptr = (const int8_t *)((const uint8_t *)B + (size_t)kk * (size_t)ldb_bytes) + j;

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
                if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
                    matrix_kernel_debug_print_prefix();
                    print_uart("  kk=");
                    matrix_kernel_debug_print_u64((uint64_t)kk);
                    print_uart(" tile_k=");
                    matrix_kernel_debug_print_u64((uint64_t)tile_k);
                    print_uart(" aptr=");
                    matrix_kernel_debug_print_ptr(aptr);
                    print_uart(" bptr=");
                    matrix_kernel_debug_print_ptr(bptr);
                    print_uart("\r\n");

                    matrix_kernel_debug_print_prefix();
                    matrix_kernel_debug_print_arr_i8("  Arow", aptr, (tile_k > 0 ? tile_k : 0));
                    matrix_kernel_debug_print_prefix();
                    matrix_kernel_debug_print_arr_i8("  Brow", bptr, (tile_k > 0 ? tile_k : 0));
                }
#endif
                matrix_kernel_mlae8_tr0(aptr, lda_bytes);
                matrix_kernel_mlbe8_tr1(bptr, ldb_bytes);
                matrix_kernel_mqma_b_acc0_tr0_tr1();
            }

            matrix_kernel_msce32_acc0(cptr, ldc_bytes);
        }
    }
    debug_delay_cycles(20);

    return 0;
}

// ---------------------------------------------------------------------------
// RVV helpers (used by quantized matmul operator below).

extern void matrix_kernel_rvv_zero_f32(float *dst, int n);

extern void matrix_kernel_rvv_zero_i32(int32_t *dst, int n);

static BARE_NO_AUTOVEC inline void matrix_kernel_rvv_zero_i32_sub(int32_t *dst, int n) {
#if MATRIX_KERNEL_DEBUG_PRINT
print_uart("=== matrix_kernel_rvv_zero_i32_sub entered ===\r\n");
#endif
    int32_t *dst_orig = dst; int n_orig = n;
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    static uint64_t s_zero_i32_sub_calls = 0;
    const uint64_t dbg_id = ++s_zero_i32_sub_calls;
    if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((dbg_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
        matrix_kernel_debug_print_prefix();
        print_uart("rvv_zero_i32_sub dst=");
        matrix_kernel_debug_print_ptr(dst);
        print_uart(" n=");
        matrix_kernel_debug_print_u64((uint64_t)n);
        print_uart("\r\n");
    }
#endif
    size_t avl = (size_t)n;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m1(avl);
        /* Create a small non-zero i32 vector then subtract it from itself to
           obtain an all-zero i32 vector. This uses integer subtract to avoid
           float/int store interactions on some targets. */
        vint32m1_t vtmp = __riscv_vmv_v_x_i32m1(12345, vl);
        vint32m1_t vzero = __riscv_vsub_vv_i32m1(vtmp, vtmp, vl);
        __riscv_vse32_v_i32m1(dst, vzero, vl);
        debug_delay_cycles(20);
        dst += vl;
        avl -= vl;
    }
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    matrix_kernel_debug_print_prefix();
    print_uart("rvv_zero_i32_sub post dst=");
    matrix_kernel_debug_print_ptr(dst_orig);
    print_uart(" n=");
    matrix_kernel_debug_print_u64((uint64_t)n_orig);
    print_uart(" vals=\r\n");
    matrix_kernel_debug_print_arr_i32("  zeroed_i32_sub", dst_orig, n_orig);
#endif
    debug_delay_cycles(20);
}

// dst[0:n] += float(src_i32[0:n]) * ws_vec[0:n] * xs
// ws_vec is loaded with strided load: ws_base + r * ws_stride_bytes.
extern void matrix_kernel_rvv_accum_i32_to_f32_ws_xs(float *dst,
                                                            const int32_t *src_i32,
                                                            const float *ws_base,
                                                            ptrdiff_t ws_stride_bytes,
                                                            float xs,
                                                            int n);

// ---------------------------------------------------------------------------
// Quantized matmul operator: W(d,n) @ x(n) -> xout(d)
// - xq: int8 vector length n
// - xs: per-group (ceil(n/gs)) scales for x
// - wq: int8 matrix row-major (d x n)
// - ws: per-group scales for w quantized over the *flattened* (d*n) array.
//
// Implementation policy:
// - No scalar per-element loops for clearing/packing/rescaling.
// - Matrix extension computes int32 dot products.
// - RVV clears output and applies ws*xs scaling + accumulation.
//
// Assumption (matches llama2 dims in this repo): n is a multiple of gs.
// If not, the ws indexing becomes non-affine across rows (because weights are
// grouped over the flattened array), and this fast path would be incorrect.
static BARE_NO_AUTOVEC inline void matrix_kernel_qmatmul_f32(float *xout,
                                             const int8_t *xq,
                                             const float *xs,
                                             const int8_t *wq,
                                             const float *ws,
                                             int n,
                                             int d,
                                             int gs,
                                             int w_row_stride) {
                                                
#if MATRIX_KERNEL_DEBUG_PRINT
//print_uart("=== matrix_kernel_qmatmul_f32 entered ===\r\n");
debug_delay_cycles(50);
#endif
#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    static uint64_t s_call_id = 0;
    const uint64_t call_id = ++s_call_id;
    if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
        matrix_kernel_debug_print_prefix();
        print_uart("qmatmul call=");
        matrix_kernel_debug_print_u64(call_id);
        print_uart("\r\n");
    }
#endif

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
    if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
        matrix_kernel_debug_print_prefix();
        print_uart("qmatmul params n=");
        matrix_kernel_debug_print_u64((uint64_t)n);
        print_uart(" d=");
        matrix_kernel_debug_print_u64((uint64_t)d);
        print_uart(" gs=");
        matrix_kernel_debug_print_u64((uint64_t)gs);
        print_uart("\r\n");
        matrix_kernel_debug_print_prefix();
        print_uart(" num_groups=");
        matrix_kernel_debug_print_u64((uint64_t)((n + gs - 1) / gs));
        print_uart(" groups_per_row=");
        matrix_kernel_debug_print_u64((uint64_t)(n / gs));
        print_uart("\r\n");
        matrix_kernel_debug_print_prefix();
        matrix_kernel_debug_print_arr_i8("xq", xq, n);
        matrix_kernel_debug_print_prefix();
        matrix_kernel_debug_print_arr_f32("xs", xs, (n + gs - 1) / gs);
    }
#endif

    // NOTE: Some Matrix implementations appear to require a minimum row stride
    // for mlce32/msce32 (e.g. 32 bytes as requested).
    // We therefore store into a padded [mblk x 1] matrix laid out as rows of 8 int32,
    // then gather the first column back into a contiguous vector for RVV.
    // NOTE: c_pack_mat / c_pack are now static file-scope buffers to avoid stack usage.

    // For calling matrix_kernel_matmul_i8_i32_abt (non-strided), we need:
    // - A packed as [mblk x count] contiguous (row stride = count)
    // - B as [8 x count] to force ldc_bytes = 8*sizeof(int32)=32 (row stride)
    //   because some Matrix implementations require >=32B row stride on C.
    enum { MK_KMAX = 32 };
    /* Use static scratch buffers to avoid large stack allocations */
    int8_t *a_pack = matrix_kernel_a_pack_buf;
    int8_t *b_pack = matrix_kernel_b_pack_buf;

    const int num_groups = (n + gs - 1) / gs;
    const int groups_per_row = n / gs;
     /* We will pack ws for each (mblk,g) into a contiguous vector, so the RVV
         accum helper always sees ws_stride_bytes==sizeof(float). */
     const ptrdiff_t ws_stride_bytes = (ptrdiff_t)sizeof(float);

    for (int i0 = 0; i0 < d; i0 += MK_MMAX) {
        const int mblk = (d - i0 < MK_MMAX) ? (d - i0) : MK_MMAX;

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
        if (((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) || (i0 == MATRIX_KERNEL_DEBUG_PRINT_I0)) {
            matrix_kernel_debug_print_prefix();
            print_uart(" i0=");
            matrix_kernel_debug_print_u64((uint64_t)i0);
            print_uart(" mblk=");
            matrix_kernel_debug_print_u64((uint64_t)mblk);
            print_uart("\r\n");
        }
#endif

        // xout block init (RVV)
        matrix_kernel_rvv_zero_f32(xout + i0, mblk);

        for (int g = 0; g < num_groups; g++) {
            const int base = g * gs;
            const int count = (base + gs <= n) ? gs : (n - base);

            // Safety: this kernel expects reasonably small per-group K.
            // If you ever set gs > MK_KMAX, bump MK_KMAX.
            if (count > MK_KMAX) {
                // Clamp to avoid stack overwrite (debug build behavior).
                // In production you should increase MK_KMAX instead.
                // NOTE: Clamping will produce incorrect results.
                // Keeping it explicit helps catch misconfiguration early.
                // (no assert/printf here to keep it freestanding-safe)
            }

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            if (((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) || (i0 == MATRIX_KERNEL_DEBUG_PRINT_I0)) {
                matrix_kernel_debug_print_prefix();
                print_uart("  g=");
                matrix_kernel_debug_print_u64((uint64_t)g);
                print_uart(" base=");
                matrix_kernel_debug_print_u64((uint64_t)base);
                print_uart("\r\n");
            }
#endif

            // int32 accum init (RVV) into padded matrix buffer
            matrix_kernel_rvv_zero_i32(c_pack_mat, mblk * 32);
            //matrix_kernel_rvv_zero_i32_sub(c_pack_mat, mblk * 8);

            // Pack A as [mblk x count] contiguous so we can call abt().
            // Source W is [d x n] row-major; we slice columns [base:base+count].
            for (int r = 0; r < mblk; r++) {
                const int8_t *src = wq + (ptrdiff_t)(i0 + r) * (ptrdiff_t)w_row_stride + (ptrdiff_t)base;
                /* pointer into packed A row for readability */
                int8_t *dst_a = a_pack + r * count;
                for (int c = 0; c < count; c++) {
                    dst_a[c] = src[c];
                }
            }

            // Replicate x segment into 8 rows so abt() uses ldc_bytes=32.
            // b_pack is [8 x count] row-major.
            const int8_t *Bseg = xq + (ptrdiff_t)base;
            /* replicate B segment into rows so abt() uses required C stride */
            for (int row = 0; row < 32; row++) {
                int8_t *dst_b = b_pack + row * count;
                for (int c = 0; c < count; c++) {
                    dst_b[c] = Bseg[c];
                }
            }

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            if (((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) || (i0 == MATRIX_KERNEL_DEBUG_PRINT_I0)) {
                matrix_kernel_debug_print_prefix();
                print_uart("   count=");
                matrix_kernel_debug_print_u64((uint64_t)count);
                print_uart(" a_pack=");
                matrix_kernel_debug_print_ptr(a_pack);
                print_uart(" b_pack=");
                matrix_kernel_debug_print_ptr(b_pack);
                print_uart(" xs=");
                matrix_kernel_debug_print_f32(xs[g]);
                print_uart("\r\n");
                matrix_kernel_debug_print_prefix();
                matrix_kernel_debug_print_arr_i8("   Bseg", Bseg, count);

                /* Debug: print packed A rows (a_pack) */
                for (int rr = 0; rr < mblk; rr++) {
                    matrix_kernel_debug_print_prefix();
                    print_uart("   a_pack row=");
                    matrix_kernel_debug_print_u64((uint64_t)rr);
                    print_uart(" ");
                    matrix_kernel_debug_print_arr_i8("a_pack_row", a_pack + rr * count, count);
                }

                /* Debug: print packed B rows (b_pack, 8 rows) */
                for (int row = 0; row < 32; row++) {
                    matrix_kernel_debug_print_prefix();
                    print_uart("   b_pack row=");
                    matrix_kernel_debug_print_u64((uint64_t)row);
                    print_uart(" ");
                    matrix_kernel_debug_print_arr_i8("b_pack_row", b_pack + row * count, count);
                }

                /* Debug: compute scalar reference src_ref[r] = dot(a_pack[r,:], Bseg[:]) and print */
                {
                    int32_t src_ref[MK_MMAX];
                    for (int rr = 0; rr < mblk; rr++) {
                        int32_t s = 0;
                        for (int cc = 0; cc < count; cc++) {
                            s += (int32_t)a_pack[rr * count + cc] * (int32_t)Bseg[cc];
                        }
                        src_ref[rr] = s;
                    }
                    /* Print src_ref in rows of 8 with prefix */
                    for (int rr = 0; rr < mblk; rr++) {
                        if ((rr % 8) == 0) {
                            if (rr != 0) print_uart("\r\n");
                            matrix_kernel_debug_print_prefix();
                            print_uart("   src_ref idx=");
                            matrix_kernel_debug_print_u64((uint64_t)rr);
                            print_uart(" vals=");
                        } else {
                            print_uart(" ");
                        }
                        matrix_kernel_debug_print_i32(src_ref[rr]);
                    }
                    print_uart("\r\n");
                }
            }
#endif

            // Use the non-strided abt() path as requested.
            // We compute C as [mblk x 8] so that C row stride is 32 bytes.
            (void)matrix_kernel_matmul_i8_i32_abt(
                a_pack,
                b_pack,
                c_pack_mat,
                mblk, 32, count);

            // Ensure Matrix Unit stores are visible to Scalar Unit
            asm volatile("fence rw,rw" ::: "memory");

            // Gather first column back into contiguous c_pack[0:mblk)
            // (avoid needing RVV strided load support for int32).
            for (int r = 0; r < mblk; r++) {
                c_pack[r] = c_pack_mat[r * 32 + 0];
            }

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            if (((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) || (i0 == MATRIX_KERNEL_DEBUG_PRINT_I0)) {
                /* Print full c_pack (no truncation) */
                /* Print c_pack in rows of 8 with prefix */
                for (int rr = 0; rr < mblk; rr++) {
                    if ((rr % 8) == 0) {
                        if (rr != 0) print_uart("\r\n");
                        matrix_kernel_debug_print_prefix();
                        print_uart("   c_pack idx=");
                        matrix_kernel_debug_print_u64((uint64_t)rr);
                        print_uart(" vals=");
                    } else {
                        print_uart(" ");
                    }
                    matrix_kernel_debug_print_i32(c_pack[rr]);
                }
                print_uart("\r\n");

                /* Recompute scalar reference and compare against c_pack */
                {
                    int32_t src_ref[MK_MMAX];
                    for (int rr = 0; rr < mblk; rr++) {
                        int32_t s = 0;
                        for (int cc = 0; cc < count; cc++) {
                            s += (int32_t)a_pack[rr * count + cc] * (int32_t)Bseg[cc];
                        }
                        src_ref[rr] = s;
                    }
                    matrix_kernel_debug_print_prefix();
                    print_uart("   cmp c_pack vs src_ref:");
                    for (int rr = 0; rr < mblk; rr++) {
                        //if (c_pack[rr] != src_ref[rr]) {
                            print_uart(" [r="); matrix_kernel_debug_print_u64((uint64_t)rr); print_uart(" got="); matrix_kernel_debug_print_i32(c_pack[rr]); print_uart(" exp="); matrix_kernel_debug_print_i32(src_ref[rr]); print_uart("]");
                        //}
                    }
                    print_uart("\r\n");
                }
            }
#endif

            // ws vector for these rows (strided), and xs scalar for this group
            // Note: `ws` here is provided as a pointer to the per-block rows starting
            // at row 0; index using r * num_groups so this works both for the
            // blocked wrapper (which passes a sub-buffer) and for the full-scale
            // call when ws is offset appropriately by the caller.
            for (int r = 0; r < mblk; r++) {
                matrix_kernel_ws_pack_buf[r] = ws[(ptrdiff_t)(i0 + r) * (ptrdiff_t)groups_per_row + (ptrdiff_t)g];
            }
            const float *ws_base = matrix_kernel_ws_pack_buf;

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
                matrix_kernel_debug_print_prefix();
                print_uart("   ws_base=");
                matrix_kernel_debug_print_ptr(ws_base);
                print_uart(" ws_stride=");
                matrix_kernel_debug_print_u64((uint64_t)ws_stride_bytes);
                print_uart("\r\n");
            }
#endif
            matrix_kernel_rvv_accum_i32_to_f32_ws_xs(
                xout + i0, c_pack, ws_base, ws_stride_bytes, xs[g], mblk);

#if MATRIX_KERNEL_DEBUG_PRINT && MATRIX_KERNEL_DEBUG_VERBOSE
            // if ((MATRIX_KERNEL_DEBUG_PRINT_EVERY <= 1) || ((call_id % (uint64_t)MATRIX_KERNEL_DEBUG_PRINT_EVERY) == 0)) {
            //     matrix_kernel_debug_print_prefix();
            //     print_uart("   xout post i0="); matrix_kernel_debug_print_u64((uint64_t)i0); print_uart(" mblk="); matrix_kernel_debug_print_u64((uint64_t)mblk); print_uart(" vals=\r\n");
            //     matrix_kernel_debug_print_prefix();
            //     matrix_kernel_debug_print_arr_f32("    xout_block", xout + i0, mblk);
            // }
#endif
        }
    }
}

#endif // MATRIX_KERNEL_H
