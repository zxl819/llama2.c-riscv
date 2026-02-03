# FPGA 裸机运行环境构建指南 (FPGA Bare-metal Build Guide)

本文档说明如何为 FPGA 硬件环境构建 Llama2 裸机推理程序。

## 典型编译命令

针对具备硬件量化加速支持的 FPGA 环境，使用以下命令进行编译：

```bash
cd 260105_chat
make -B rvbareclang \
    RV_BARE_APP=./run_bare_hw_scalar_clang_quant.c \
    RV_BARE_NAME=run_bare_hw_scalar_clang_quant \
    BARE_PLATFORM=fpga \
    PRINT_WAY=uart \
    BARE_EMBED_BLOBS=1
```

## 参数详细说明

- **`make -B`**: 强制重新构建所有目标，确保所有配置更改（如宏定义切换）都能生效。
- **`rvbareclang`**: 使用基于 Clang 的交叉编译工具链。该目标会自动链接 Matrix 相关的硬件内核（如 `matrix_kernel_noblk.o` 和 `matrix_kernel_noblk_CT.o`）以及 RVV 累加内核。
- **`RV_BARE_APP`**: 指定主程序入口文件。在此示例中为 `run_bare_hw_scalar_clang_quant.c`（标量量化版本）。
    - 注意：如果文件位于父目录，请使用 `RV_BARE_APP=../filename.c`。
    - 常用变体还包括 `run_bare_hw_matrix_clang_quant_chat.c` 等聊天交互版本。
- **`RV_BARE_NAME`**: 指定生成的二进制文件名（不含扩展名）。输出将存放在 `build/bare/` 下。
- **`BARE_PLATFORM=fpga`**: 关键参数。
    - 切换起始代码至 `crt_uart.S`（初始化串口相关的硬件状态）。
    - 切换系统调用内核至 `bare_syscalls_uart.c`（将 `printf` 等输出重定向到硬件 UART 寄存器）。
- **`PRINT_WAY=uart`**: 明确指定打印输出路径为硬件 UART，而非 Spike 模拟器的 HTIF 接口。
- **`BARE_EMBED_BLOBS=1`**: 静态链接模型数据。
    - 将 `stories15M.bin` (或指定的 `MODEL`) 和 `tokenizer.bin` 转换为汇编代码并嵌入到 ELF 文件的 `.model_blob` 和 `.tokenizer_blob` 段。
    - 这样编译出的程序是自包含的，可以直接刷入 FPGA 内存运行，无需文件系统支持。

## 编译输出

编译完成后，相关文件将生成在 `build/bare/` 目录下：

1. **`run_bare_hw_scalar_clang_quant.elf`**: 包含符号调试信息的 ELF 格式文件。
2. **`run_bare_hw_scalar_clang_quant.bin`**: 用于烧录或加载到 FPGA 内存的纯二进制镜像。

## 反汇编与调试

若需要检查硬件指令生成的正确性（例如确认是否调用了 Matrix 指令或 RVV 指令），可以运行：

```bash
make dump RV_BARE_NAME=run_bare_hw_scalar_clang_quant
```

这将生成 `build/bare/run_bare_hw_scalar_clang_quant.dump` 文件。
