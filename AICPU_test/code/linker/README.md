# AICPU_test 编译说明

本目录（`AICPU_test/code/linker/`）包含所有裸机测试用例的构建系统，所有依赖文件均自包含于 `AICPU_test/` 目录下，无需依赖仓库根目录的其他源文件。

---

## 目录结构

```
AICPU_test/
├── code/
│   ├── matmul_matrix_inline_16x16.c   # 16×16 int8 矩阵乘 demo（AME inline-asm）
│   ├── softmax.c                      # Softmax demo
│   └── linker/                        # ← 本目录（构建系统所在位置）
│       ├── Makefile
│       ├── test_compact.ld            # 链接脚本
│       ├── crt_uart.S                 # 启动代码
│       ├── bare_syscalls_uart.c       # 裸机 syscall（memcpy/memset/exit/abort）
│       ├── uart.c / uart.h            # 硬件 UART 驱动
│       ├── uart_helper.c              # 打印辅助（被 app 直接 #include）
│       ├── matrix_kernel_1230.h       # 矩阵内核头文件
│       ├── matrix_kernel_matmul.c     # 矩阵乘实现
│       ├── matrix_kernel_noblk_1231.h
│       ├── matrix_kernel_noblk.c      # 无分块矩阵乘实现
│       ├── encoding.h                 # RISC-V CSR 编码
│       └── build/                     # 编译输出目录（自动创建）
```

---

## 环境要求

| 工具 | 说明 |
|------|------|
| `riscv64-unknown-elf-gcc` | GCC 工具链（gcc 模式必须） |
| `riscv64-unknown-elf-objcopy/objdump` | 二进制工具 |
| `~/newllvm/test_llvm/build/bin/clang` | 自定义 LLVM/Clang（clang 模式必须） |
| `spike` | RISC-V 模拟器（`make run` 时使用） |

**目标架构**：`rv64gcv_zfh_zvfh_zvl512b`，ABI：`lp64d`，内存模型：`medany`

---



## 编译命令

所有命令均在 `AICPU_test/code/linker/` 目录下执行。

### 使用 Clang 编译

```bash
# 方式一：使用专用目标（推荐）
make rvbareclang APP=matmul_matrix_inline_16x16

# 方式二：手动指定 TOOLCHAIN
make rvbare TOOLCHAIN=clang APP=matmul_matrix_inline_16x16
```

### 可用 APP 列表

| APP 名称 | 说明 | MATRIX_SUPPORT |
|----------|------|----------------|
| `matmul_matrix_inline_16x16` | 16×16 int8 矩阵乘，AME inline-asm（默认） | 不需要 |
| `softmax` | Softmax 实现 | 不需要 |

---

## 其他 Make 目标

```bash
# 生成反汇编 dump 文件
make dump APP=matmul_matrix_inline_16x16

# 查看当前配置信息
make info

# 清除构建输出
make clean
```

---

## 常用变量覆盖

```bash
# 自定义 Clang 路径
make rvbareclang RV_BARE_CLANG=/path/to/your/clang
```

---

## 编译输出

编译产物位于 `build/` 目录：

| 文件 | 说明 |
|------|------|
| `build/<APP>.elf` | ELF 可执行文件（含调试信息） |
| `build/<APP>.bin` | 纯二进制镜像 |
| `build/<APP>.dump` | 反汇编文件（`make dump` 后生成） |

---

## 调试说明

- **非法指令异常**：通常是 RVV/Matrix 状态未启用。确认运行环境（Spike/硬件）已使能 VS、XS、FS 状态。
- **`libgcc.a` 找不到**：clang 的 `ld.lld` 不支持 `-lgcc` 标志，需传入 `.a` 绝对路径。确保已运行 `make setup` 或 `libgcc.a` 存在于本目录。
- **查看 trap 信息**：裸机 `handle_trap` 会通过 UART 打印 `mcause`/`mepc`/`mtval` 及附近指令字，可直接用于定位故障 opcode。
