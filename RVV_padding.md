# RVV_padding.c 说明（静态 padding 优先）

本文档面向 `llama2.c-riscv/RVV_padding.c`，重点说明 **freestanding/bare-metal** 场景下默认启用的 **静态缓冲区 padding**：它不依赖 `malloc/calloc/free`，通过内部 `static` 缓冲区返回补齐后的数据指针。

## 1. 选择哪种 padding 后端

该文件支持两种后端：

- **静态缓冲区模式（推荐用于 bare-metal）**：返回指向内部静态数组的指针。
- **动态分配模式（hosted 有 libc 时常用）**：走系统 `malloc/calloc/free`（或可选的简易 bare bump allocator）。

后端选择逻辑由宏控制：

- `RVV_PADDING_STATIC`
  - 未定义时：
    - 若 `__STDC_HOSTED__` 为 0（freestanding），默认 `RVV_PADDING_STATIC=1`（静态模式）。
    - 若 hosted（有 libc），默认 `RVV_PADDING_STATIC=0`（malloc 模式）。
  - 你可以在编译时强制：
    - `-DRVV_PADDING_STATIC=1` 强制静态模式
    - `-DRVV_PADDING_STATIC=0` 强制 malloc 模式

## 2. 静态 padding 模式（RVV_PADDING_STATIC=1）

### 2.1 内部静态缓冲区与限制

静态模式下，文件内部维护两块缓冲区：

- `g_pad_1d_buf`：给一维数组 padding 使用
- `g_pad_2d_buf`：给二维矩阵 padding 使用

容量可配置：

- `RVV_PADDING_STATIC_1D_BYTES`：默认 `8 * 1024`
- `RVV_PADDING_STATIC_2D_BYTES`：默认 `64 * 1024`
- `BARE_ALLOC_ALIGN`：默认对齐 `64` 字节（用于 `__attribute__((aligned()))`）

重要限制（静态模式的设计取舍）：

- **同一时间每种缓冲区只能“占用一次”**：
  - 1D 和 2D 各自都有一个 `in_use` 标志（`g_pad_1d_in_use` / `g_pad_2d_in_use`）。
  - 在未释放前再次申请，会返回 `NULL`。
- **超过缓冲区上限会返回 `NULL`**。
- **不线程安全/不可重入**：适合裸机单线程或受控调用序列。

> 直观理解：静态模式不是“通用分配器”，而是为了 bare-metal 下“少依赖、可预测”的临时补齐缓冲。

### 2.2 对外 API（静态模式同样适用）

该文件对外主要提供 4 个函数：

#### A) `PaddedArray1D pad_array_1d(...)`

功能：

- 为一维数组（向量）创建一个“补齐到 `padded_len`”的新缓冲区：
  - 前 `original_len` 个元素从 `original_array` 复制。
  - 剩余部分填 0。

函数签名：

- `PaddedArray1D pad_array_1d(const void* original_array, size_t original_len, size_t element_size, size_t padded_len)`

参数说明：

- `original_array`：原始数据指针（不能为空；该函数不支持 `NULL` 代表“只分配并清零”，这点与 2D 不同）
- `original_len`：原始元素个数
- `element_size`：单个元素字节数，例如 `sizeof(float)`
- `padded_len`：目标补齐后的元素个数（必须 `>= original_len`）

返回值：`PaddedArray1D`

- `data`：补齐后数据指针；失败时为 `NULL`
- `original_len`：原始长度
- `padded_len`：补齐后长度
- `element_size`：元素大小

失败条件（返回 `data==NULL`）：

- `padded_len < original_len`
- 申请大小超过 `RVV_PADDING_STATIC_1D_BYTES`（静态模式）
- 1D 缓冲区已被占用（静态模式）
- malloc 失败（动态模式）

#### B) `PaddedArray2D pad_matrix_2d(...)`

功能：

- 为二维矩阵创建一个“补齐列到 `padded_cols`”的新缓冲区：
  - 每行前 `original_cols` 个元素从 `original_matrix` 拷贝。
  - 每行剩余列填 0。

函数签名：

- `PaddedArray2D pad_matrix_2d(const void* original_matrix, size_t rows, size_t original_cols, size_t element_size, size_t padded_cols)`

参数说明：

- `original_matrix`：原始矩阵数据指针
  - **允许为 `NULL`**：表示“只分配并清零整块 `rows * padded_cols` 缓冲区”（便于直接创建零矩阵）。
- `rows`：矩阵行数
- `original_cols`：原始列数
- `element_size`：元素字节数（例如 `sizeof(float)`）
- `padded_cols`：目标补齐后的列数（必须 `>= original_cols`）

返回值：`PaddedArray2D`

- `data`：补齐后矩阵指针；失败时为 `NULL`
- `rows`：行数
- `original_cols`：原始列数
- `padded_cols`：补齐后列数
- `element_size`：元素大小

失败条件（返回 `data==NULL`）：

- `padded_cols < original_cols`
- 申请大小超过 `RVV_PADDING_STATIC_2D_BYTES`（静态模式）
- 2D 缓冲区已被占用（静态模式）
- malloc 失败（动态模式）

#### C) `void free_padded_array_1d(PaddedArray1D* array)`

功能：

- 释放/归还 `pad_array_1d` 的结果。

静态模式行为：

- 不会真正 free 内存；只是把 `g_pad_1d_in_use=0`，允许下一次 1D 申请。

动态模式行为：

- 调用 `free(array->data)`。

注意：

- 该函数会把 `array->data` 置 `NULL` 以避免悬垂指针。

#### D) `void free_padded_matrix_2d(PaddedArray2D* matrix)`

功能：

- 释放/归还 `pad_matrix_2d` 的结果。

静态模式行为：

- 不会真正 free 内存；只是把 `g_pad_2d_in_use=0`。

动态模式行为：

- 调用 `free(matrix->data)`。

同样会把 `matrix->data` 置 `NULL`。

## 3. 动态分配模式（RVV_PADDING_STATIC=0）补充

当 `RVV_PADDING_STATIC=0` 时：

- hosted 环境默认使用系统 `malloc/calloc/free`。
- freestanding 环境若没有提供 libc：
  - 可以选择启用文件内的“简易 bump allocator”（需要定义 `BARE_HEAP_BYTES`）。
  - 若你的工程已经自己提供了 `malloc/calloc/free`，可以定义 `RVV_PADDING_NO_ALLOCATOR` 避免重复定义。

## 4. 典型用法（静态模式）

一维：

- 调用 `pad_array_1d(src, n, sizeof(float), padded_n)` 得到 `PaddedArray1D`。
- 使用 `result.data` 作为补齐后的连续内存。
- 用完必须调用 `free_padded_array_1d(&result)` 归还静态缓冲区占用。

二维：

- 调用 `pad_matrix_2d(src, rows, cols, sizeof(float), padded_cols)`。
- 若只想拿到一个全 0 的补齐矩阵，可传 `original_matrix=NULL`。
- 用完调用 `free_padded_matrix_2d(&result)`。

## 5. 常见坑与建议

- 静态模式下 **1D/2D 各自只能同时持有一个结果**；不要在未 free 前再次申请同类型 padding。
- 如果你有并行/嵌套调用需求，考虑：
  - 增大静态缓冲并实现多块池化管理，或
  - 切换到动态分配模式（hosted/malloc）。
- `element_size` 必须与你真实数据类型一致，否则会导致拷贝/清零长度错误。
