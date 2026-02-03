# RISC-V LLM 推理加速与优化工作总结

## 1. 项目概览
本项目旨在基于 RISC-V 架构（集成自定义矩阵扩展 AME 与 向量扩展 RVV）的 FPGA 原型平台，实现 `llama2.c` 的裸机（Bare-metal）高效推理。工作涵盖了从标量实现到硬件加速算子的移植、性能分析以及针对性的算子优化。

## 2. 版本演进与特点

项目代码经历了三个主要阶段的迭代，分别对应不同的目录版本：

### **Phase 1: 260105_chat (功能验证与Chat实现)**
*   **目标**: 完成 LLM Chat 模式在 RISC-V 裸机环境下的跑通。
*   **主要工作**:
    *   移植 `run.c` 到裸机环境 (`run_bare_*.c`)。
    *   实现基础的 UART 驱动与 minimal libc (`bare_syscalls_uart.c`)。
    *   提供了标量 (`scalar`) 和矩阵 (`matrix`) 两个版本的推理入口。
    *   使用DDR存储权重、激活等参数。
    *   尝试初步集成 Softmax RVV 算子。【Undo】
*   **特点**: 
    *   侧重功能正确性，支持交互式 Chat 模式。


### **Phase 2: 260113_perf (性能分析与低精度探索)**
*   **目标**: 建立性能基准，探索 FP8 低精度数据类型。
*   **主要工作**:
    *   修改ld文件，使用22M的SRAM存储权重、激活等参数。
    *   引入 FP8 数据定义 (`data_16x16_fp8.h`, `fp8_e4m3fn.h`)，为后续低精度量化做准备。
*   **特点**: 
    *   包含详细的性能打点数据（见 `build/perf/`），开始引入实验性的 FP8 矩阵乘实现。
    *   添加性能 Profiling 代码 (`run_bare_..._perf.c`)，统计各层算子耗时。
    *   包含 `check_tok.py` 等辅助脚本用于查验 Token 生成正确性。


### **Phase 3: 260123_matmulOP (算子深度优化与调试)**
*   **目标**: 解决性能瓶颈，深度优化矩阵乘算子。
*   **当前状态**: **Active / Debugging**
*   **主要工作**:
    *   **矩阵内核优化 (`matrix_kernel_noblk.c`、`matrix_kernel_noblk_CT.c`)**: 
        *   **重大修复**: 移除了核心计算循环中的大型 VLA（变长数组）栈分配，为解决 Cache Coherence 和 栈溢出风险， 改为C_pack为全局静态变量。
    *   【Undo】测试Uncache端能否正确使用
*   **特点**: 
    *   针对 AME 硬件特性进行了深度适配，目前正集中集成进入模型但结果性能下降的问题。


## 3. 关键优化技术点总结

### 3.1 内存管理与 Cache 一致性
*   **问题**: FPGA 原型上 CPU Cache 与 加速器（AME）DMA 存在一致性挑战。
*   **优化**: 
    *   在 `test_compact.ld` 中划分专用 Section。
    *   【Undo】将涉及加速器读写的大型中间 Buffer（如量化后的 `c_pack`）声明为 `__attribute__((section(".uncached_buffer")))`，确保数据直接落盘 DDR，避免 Flush/Invalidate 开销及潜在的一致性 Bug。

### 3.2 矩阵乘法 (Matmul) 内核优化
*   **原始方案**: 使用栈上 VLA 数组临时存储量化参数，导致栈空间压力大且难以控制缓存行为。
*   **优化方案**: 
    *   **寄存器存储优化：**矩阵乘`C_pack`优化为转置存储。
    *   **Static Global Buffer**: 将 `c_pack` 等临时缓冲区移至全局静态区。
    *   **Batch 处理**: 针对 Transpose 和 Batch Matmul 进行了循环展开与流水线优化。

### 3.3 调试基础设施
*   **UART 增强**: 实现了 `print_uart_int_hex`，便于直接查看物理地址和寄存器原始值。
*   **Trace System**: 在 `matrix_kernel` 层级增加了细粒度的执行流打印（Tag: `[debug_trace]`），协助定位指令级 Hang。

## 4. AICPU_Test 测试包构建
整理了独立的 `AICPU_test` 目录，用于算子级回归测试：
*   包含完整的 `Makefile` 和 `linker` 脚本。
*   提供 `Softmax` (RVV) 和 `Matmul` (AME/Int8) 的独立测试 Case (`test_*.c`)。
*   文档化了内存布局和编译流程，便于团队协作与交付。

## 5. 下一步计划
1. **验证 Fix**: 确认将 `c_pack` 移入 Uncached 区域后，是否彻底解决了 Prefill 阶段的 Hang 问题。

2. **性能回归**: 在矩阵乘算子优化稳定性修复后，重新运行 `260113_perf` 中的性能测试，对比优化前后的 Cycle 数。

3. **FP8 落地**: 继续推进 Phase 2 中开启的 FP8 算子集成，进一步提升带宽利用率。

4. **SRAM使用：**继续推进debug 权重激活全部使用SRAM后的性能。

   
