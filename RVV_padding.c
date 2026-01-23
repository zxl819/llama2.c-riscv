// NOTE: This file is often included into freestanding (bare-metal) builds where
// a full libc (and headers like <stdlib.h>) may not be available.
//
// Padding策略：
// - Hosted（有 libc）默认使用系统 malloc/calloc/free。
// - Freestanding（bare-metal）默认使用“静态数组补齐”的方式：
//   pad_array_1d/pad_matrix_2d 从内部 static 缓冲区返回指针，不依赖 malloc。
//   若申请尺寸超过静态缓冲上限，会返回 data==NULL。

#if defined(__STDC_HOSTED__) && (__STDC_HOSTED__)
    #include <stdlib.h>
    // In hosted builds, prefer the system allocator by default.
    #ifndef RVV_PADDING_NO_ALLOCATOR
        #define RVV_PADDING_NO_ALLOCATOR 1
    #endif
#else
    #include <stddef.h>
    // Declarations for freestanding builds if needed.
    void exit(int);
#endif

#include <string.h>
#include <stdint.h> // for uintptr_t

// Default alignment for the optional bare allocator (when enabled).
#ifndef BARE_ALLOC_ALIGN
#define BARE_ALLOC_ALIGN 64
#endif

// ---------------------------------------------------------------------------
// Select padding backend
// - Define RVV_PADDING_STATIC=0 to force malloc path even in freestanding.
// - Define RVV_PADDING_STATIC=1 to force static-buffer path.
#ifndef RVV_PADDING_STATIC
#if !defined(__STDC_HOSTED__) || !(__STDC_HOSTED__)
#define RVV_PADDING_STATIC 1
#else
#define RVV_PADDING_STATIC 0
#endif
#endif

// Static-buffer sizing (only used when RVV_PADDING_STATIC=1)
// Tune these as needed; functions return NULL when exceeding the limit.
#ifndef RVV_PADDING_STATIC_1D_BYTES
#define RVV_PADDING_STATIC_1D_BYTES (8 * 1024)
#endif
#ifndef RVV_PADDING_STATIC_2D_BYTES
#define RVV_PADDING_STATIC_2D_BYTES (64 * 1024)
#endif

#if RVV_PADDING_STATIC
__attribute__((aligned(BARE_ALLOC_ALIGN))) static unsigned char g_pad_1d_buf[RVV_PADDING_STATIC_1D_BYTES];
__attribute__((aligned(BARE_ALLOC_ALIGN))) static unsigned char g_pad_2d_buf[RVV_PADDING_STATIC_2D_BYTES];
static int g_pad_1d_in_use = 0;
static int g_pad_2d_in_use = 0;

static void *rvv_pad_static_alloc_1d(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_pad_1d_in_use) return NULL;
    if (bytes > (size_t)RVV_PADDING_STATIC_1D_BYTES) return NULL;
    g_pad_1d_in_use = 1;
    return (void*)g_pad_1d_buf;
}

static void *rvv_pad_static_alloc_2d(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_pad_2d_in_use) return NULL;
    if (bytes > (size_t)RVV_PADDING_STATIC_2D_BYTES) return NULL;
    g_pad_2d_in_use = 1;
    return (void*)g_pad_2d_buf;
}
#endif

// ---------------------------------------------------------------------------
// Optional tiny bump allocator for bare-metal use (legacy)
// If you want to keep using malloc/calloc/free in freestanding builds, you can:
//   - set RVV_PADDING_STATIC=0
//   - define BARE_HEAP_BYTES and ensure no other malloc is linked
// If the including TU already provides malloc/calloc/free, define RVV_PADDING_NO_ALLOCATOR.
#if !RVV_PADDING_STATIC
#ifndef RVV_PADDING_NO_ALLOCATOR
static unsigned char bare_heap[BARE_HEAP_BYTES];
static size_t bare_heap_offset = 0;

static void panic(const char *msg) {
    (void)msg;
    exit(1);
}

static void *bare_alloc(size_t size, size_t alignment) {
    if (alignment == 0) alignment = BARE_ALLOC_ALIGN;
    size_t misalignment = bare_heap_offset % alignment;
    if (misalignment != 0) bare_heap_offset += alignment - misalignment;
    if (bare_heap_offset + size > BARE_HEAP_BYTES) panic("bare heap exhausted");
    void *ptr = &bare_heap[bare_heap_offset];
    bare_heap_offset += size;
    return ptr;
}

void *malloc(size_t size) {
    if (size == 0) size = 1;
    return bare_alloc(size, BARE_ALLOC_ALIGN);
}

void *calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return bare_alloc(1, BARE_ALLOC_ALIGN);
    size_t total = count * size;
    void *ptr = bare_alloc(total, BARE_ALLOC_ALIGN);
    unsigned char *p = (unsigned char *)ptr;
    for (size_t i = 0; i < total; i++) {
        p[i] = 0;
    }
    return ptr;
}

void free(void *ptr) { (void)ptr; }
#endif
#endif

// --- 一维数组（向量）的填充结果 ---
typedef struct {
    void* data;          // 指向填充后数据的指针
    size_t original_len;  // 原始长度
    size_t padded_len;   // 填充后的长度
    size_t element_size;  // 每个元素的大小 (例如 sizeof(float))
} PaddedArray1D;

// --- 二维数组（矩阵）的填充结果 ---
typedef struct {
    void* data;          // 指向填充后数据的指针
    size_t rows;         // 行数
    size_t original_cols; // 原始列数
    size_t padded_cols;  // 填充后的列数
    size_t element_size;  // 每个元素的大小
} PaddedArray2D;

/**
 * @brief 为一维数组进行零填充
 * 
 * @param original_array 指向原始数组的指针
 * @param original_len 原始数组的长度
 * @param element_size 每个元素的字节大小
 * @param padded_len 填充后的目标长度
 * @return PaddedArray1D 包含填充后数据和信息的结构体。
 *         如果失败，其 data 成员将为 NULL。
 */
__attribute__((optnone)) PaddedArray1D pad_array_1d(const void* original_array, size_t original_len, size_t element_size, size_t padded_len) {
    PaddedArray1D result;
    result.data = NULL;
    result.original_len = 0;
    result.padded_len = 0;
    result.element_size = 0;

    if (padded_len < original_len) {
        // 填充长度不能小于原始长度
        return result;
    }

    // 分配填充后的内存空间
#if RVV_PADDING_STATIC
    result.data = rvv_pad_static_alloc_1d(padded_len * element_size);
#else
    result.data = malloc(padded_len * element_size);
#endif
    if (!result.data) {
        return result; // 内存分配失败
    }

    // 复制原始数据
    if (original_len > 0) {
        memcpy(result.data, original_array, original_len * element_size);
    }

    // 将剩余部分用零填充
    if (padded_len > original_len) {
        // 使用 uintptr_t 进行指针运算以确保安全
        memset((uint8_t*)result.data + original_len * element_size, 0, (padded_len - original_len) * element_size);
    }

    result.original_len = original_len;
    result.padded_len = padded_len;
    result.element_size = element_size;

    return result;
}

/**
 * @brief 为二维数组（矩阵）进行零填充
 * 
 * @param original_matrix 指向原始矩阵的指针
 * @param rows 矩阵的行数
 * @param original_cols 矩阵的原始列数
 * @param element_size 每个元素的字节大小
 * @param padded_cols 填充后的目标列数
 * @return PaddedArray2D 包含填充后数据和信息的结构体。
 *         如果失败，其 data 成员将为 NULL。
 */
__attribute__((optnone)) PaddedArray2D pad_matrix_2d(const void* original_matrix, size_t rows, size_t original_cols, size_t element_size, size_t padded_cols) {
    PaddedArray2D result;
    result.data = NULL;
    result.rows = 0;
    result.original_cols = 0;
    result.padded_cols = 0;
    result.element_size = 0;

    if (padded_cols < original_cols) {
        return result;
    }

    // 分配填充后的内存空间
#if RVV_PADDING_STATIC
    result.data = rvv_pad_static_alloc_2d(rows * padded_cols * element_size);
#else
    result.data = malloc(rows * padded_cols * element_size);
#endif
    if (!result.data) {
        return result;
    }

    // Allow NULL source to mean "allocate and zero-init"
    if (original_matrix == NULL) {
        memset(result.data, 0, rows * padded_cols * element_size);
        result.rows = rows;
        result.original_cols = original_cols;
        result.padded_cols = padded_cols;
        result.element_size = element_size;
        return result;
    }

    const uint8_t* src_row = (const uint8_t*)original_matrix;
    uint8_t* dest_row = (uint8_t*)result.data;

    for (size_t i = 0; i < rows; ++i) {
        // 1. 复制当前行的原始数据
        memcpy(dest_row, src_row, original_cols * element_size);

        // 2. 填充当前行的剩余部分
        if (padded_cols > original_cols) {
            memset(dest_row + original_cols * element_size, 0, (padded_cols - original_cols) * element_size);
        }

        // 移动到下一行
        src_row += original_cols * element_size;
        dest_row += padded_cols * element_size;
    }

    result.rows = rows;
    result.original_cols = original_cols;
    result.padded_cols = padded_cols;
    result.element_size = element_size;

    return result;
}

void free_padded_array_1d(PaddedArray1D* array) {
    if (array && array->data) {
#if RVV_PADDING_STATIC
        g_pad_1d_in_use = 0;
#else
        free(array->data);
#endif
        array->data = NULL; // 防止悬垂指针
    }
}

void free_padded_matrix_2d(PaddedArray2D* matrix) {
    if (matrix && matrix->data) {
#if RVV_PADDING_STATIC
        g_pad_2d_in_use = 0;
#else
        free(matrix->data);
#endif
        matrix->data = NULL; // 防止悬垂指针
    }
}