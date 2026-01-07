# Copilot / AI agent instructions for llama2.c-riscv

目的：帮助 AI 编码代理快速上手本仓库的结构、构建与调试惯例，以及与硬件/矩阵内核的集成点。

- **代码大体架构**：
  - 推理驱动：`run_bare_hw_matrix_clang_quant.c`（bare-metal、嵌入模型/分词器二进制、UART 输出）和 `run.c`（用户态/主机版）是主要运行入口。
  - 矩阵加速：矩阵/量化内核分散在 `matrix_kernel_*.h` / `matrix_kernel_noblk_*.h`，这些头文件封装了 RVV/Matrix 指令路径（int8 x int8 -> int32）。
  - 模型/分词器：模型二进制通过 Makefile 的 rvbare 目标由 objcopy 嵌入，链接符号为 `MODEL_BIN_START/END` 和 `TOKENIZER_BIN_START/END`（见 `run_bare_hw_matrix_clang_quant.c` 文件头注释）。

- **关键数据流**：
  - 加载：嵌入 blob -> 读取 `Config` 头 -> memory-map 到 `TransformerWeights`（`memory_map_weights`）。
  - 量化：初始化时使用 `quantize_const` 将 matmul 权重转为 `QuantizedTensor`；推理时临时对激活使用 `quantize`。
  - 计算：前向使用 `matmul_q`（封装为 `matrix_kernel_qmatmul_f32_noblk`）来利用矩阵/向量加速路径。

- **常见修改点与约定**：
  - 量化组大小用 `GS` 宏控制（默认 32），可通过编译时宏覆盖：`CFLAGS+=-DGS=64`。
  - Bare-metal 可配置宏：`BARE_PROMPT`, `BARE_STEPS`, `BARE_TEMPERATURE`, `BARE_TOPP`, `BARE_SEED`, `BARE_HEAP_BYTES`, `BARE_CPU_HZ`, `BARE_USE_CYCLE_COUNTER`。
  - 禁止 Clang 自动向量化的注解为 `BARE_NO_AUTOVEC`（见文件顶部条件宏）。
  - 内存分配：使用 repo 内的微型 bump allocator（`bare_heap`），注意 `BARE_HEAP_BYTES` 大小会影响可分配内存。

- **构建 / 调试 工作流**（可在 Makefile 中找到详细目标）
  - 构建 bare 版本：参见 Makefile 的 `rvbare` 目标（会把 MODEL/TOKENIZER 嵌入 ELF）。常见修改示例：
    - 修改量化组：`make rvbare CFLAGS+=-DGS=64`
    - 替换嵌入二进制：在 Makefile 调用处传入或修改 `MODEL_BIN` / `TOKENIZER_BIN`，或参考 `run_bare_hw_matrix_clang_quant.c` 的注释。
  - 在 Spike/模拟器上运行：bare 程序通过 UART 输出与 tohost/fromhost 协议交互；若出现 trap，`handle_trap` 会通过 UART 打印 mcause/mepc/mtval 及附近指令字，作为首选诊断信息源。

- **调试要点**
  - 若遇到“非法指令”，很可能是未启用 RVV/Matrix 状态：调用 `enable_rvv_state()` 或在运行环境启用 VS、XS、FS 状态（见 `enable_rvv_state` 实现）。
  - `BARE_USE_CYCLE_COUNTER` 打开后会使用 `rdcycle()` 记录循环和 TTFT（first-matmul），在 perf/性能调优时有用。
  - `handle_trap` 的内存转储可直接用于定位在 Spike 中的故障 opcode。

- **代码风格 / 约定示例**
  - 将重量矩阵按 `(n,d)` 视角组织（见 `memory_map_weights`），多处直接使用 `dim`, `hidden_dim`, `kv_dim`。
  - 量化结构：`typedef struct { int8_t *q; float *s; } QuantizedTensor;`，`s` 长度为 `ceil(n/GS)`。
  - 不在库函数中依赖 libc（bare 版本实现了 `malloc/calloc/free/memcpy/memset/exit` 的小型替代）。

- **定位关键文件**（优先阅读顺序）
  - [run_bare_hw_matrix_clang_quant.c](run_bare_hw_matrix_clang_quant.c) — bare-metal 推理流程、量化与矩阵调用示例。
  - matrix_kernel_1230.h / matrix_kernel_noblk_1231.h — 矩阵内核接口（加速路径实现点）。
  - Makefile — 构建目标、嵌入步骤（rvbare）和编译宏示例。
  - README.md — 项目整体说明与高层使用示例。

- **对 AI 代理的行为准则（针对本仓库）**
  - 优先保守更改：在修改矩阵内核或量化逻辑前，保留原接口并添加宏开关或新函数，避免破坏硬件加速路径。
  - 对性能相关变更提供基准：在更改 `GS`、数据布局或内核代码时，添加短的 micro-benchmark（或示例 run）说明性能影响。
  - 代码变更附带测试说明：说明如何用 `make rvbare` 构造嵌入二进制并在模拟器/硬件上验证（参考 `handle_trap` 输出用于调试）。

请审阅此草稿并指出需要补充的部分（例如 Makefile 中的精确参数形式或你希望列出的模拟器/运行命令）。
